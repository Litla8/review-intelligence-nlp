"""Step 6 pipeline: fine-tune DistilBERT on the same split as Step 5 and compare it with the classical models.

Run from the project root::

    python -m reviews.train_transformer --sample    # 30,000 reviews, a few minutes, to check the setup
    python -m reviews.train_transformer             # all reviews, roughly 20-40 minutes on an RTX 5070
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import joblib
import numpy as np
import pandas as pd

from reviews import classical, plots, transformer
from reviews.config import ProjectConfig, TransformerConfig
from reviews.logging_utils import configure_logging
from reviews.metrics import evaluate_classifier, time_inference, worth_the_cost

LOGGER = logging.getLogger(__name__)
MODEL_NAME = "distilbert_finetuned"


@dataclass
class FitOutput:
    """What a training run hands back: logits for validation and test, the time it took, and a text predictor for timing."""

    val_logits: np.ndarray
    test_logits: np.ndarray
    train_seconds: float
    predict: Callable[[Sequence[str]], Sequence]


FitFn = Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame, TransformerConfig, Path], FitOutput]


def _accepts(callable_: Any, name: str) -> bool:
    """True if ``callable_`` takes a parameter called ``name`` (Hugging Face renamed several between versions)."""
    return name in inspect.signature(callable_).parameters


def fit_distilbert(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, cfg: TransformerConfig,
                   model_dir: Path) -> FitOutput:
    """Fine-tune with the Hugging Face Trainer on the raw review text and save the best checkpoint to ``model_dir``.

    Raises:
        ImportError: If torch / transformers are missing.
        RuntimeError: If PyTorch cannot see a GPU.
    """
    try:
        import torch
        from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer,
                                  TrainingArguments, set_seed)
    except ImportError as exc:
        raise ImportError("Step 6 needs torch and transformers. Install torch first (CUDA 12.8 build for RTX 50-series): "
                          "pip install torch --index-url https://download.pytorch.org/whl/cu128 ; "
                          "then: pip install transformers accelerate") from exc
    cuda = torch.cuda.is_available()
    LOGGER.info(transformer.gpu_message(cuda, torch.cuda.get_device_name(0) if cuda else None, torch.__version__))
    if not cuda:
        raise RuntimeError(transformer.gpu_message(False, None, torch.__version__))
    set_seed(classical.SEED)
    classes = ["negative", "positive"]
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=2, id2label=dict(enumerate(classes)), label2id={c: i for i, c in enumerate(classes)})

    def encode(frame: pd.DataFrame) -> transformer.EncodedDataset:
        enc = tokenizer(frame["Text"].astype(str).tolist(), truncation=True, max_length=cfg.max_length)
        return transformer.EncodedDataset({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]},
                                          transformer.label_ids(frame[classical.LABEL], classes))

    started = time.perf_counter()
    train_ds, val_ds, test_ds = encode(train), encode(val), encode(test)
    LOGGER.info("tokenised %d reviews in %.0fs", len(train) + len(val) + len(test), time.perf_counter() - started)

    steps_per_epoch = -(-len(train_ds) // cfg.train_batch_size)
    strategy = {("eval_strategy" if _accepts(TrainingArguments, "eval_strategy") else "evaluation_strategy"): "epoch"}
    args = TrainingArguments(
        output_dir=str(model_dir / "checkpoints"), num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.train_batch_size, per_device_eval_batch_size=cfg.eval_batch_size,
        learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay,
        warmup_steps=int(cfg.warmup_fraction * steps_per_epoch * cfg.epochs),
        bf16=bool(cfg.bf16 and torch.cuda.is_bf16_supported()), save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
        logging_steps=cfg.logging_steps, seed=classical.SEED, report_to="none",
        dataloader_num_workers=0,  # 0 avoids multiprocessing problems on Windows
        **strategy)
    tokenizer_arg = "processing_class" if _accepts(Trainer.__init__, "processing_class") else "tokenizer"
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=transformer.compute_metrics,
                      **{tokenizer_arg: tokenizer})
    started = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - started
    LOGGER.info("training took %.1f minutes", train_seconds / 60)
    val_logits, test_logits = trainer.predict(val_ds).predictions, trainer.predict(test_ds).predictions

    best_model = trainer.model
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    use_bf16 = bool(args.bf16)

    def predict(texts: Sequence[str]) -> np.ndarray:
        """Predicted class ids for raw review texts (tokenising included, so timing is fair against TF-IDF models)."""
        best_model.eval()
        outputs = []
        with torch.no_grad():
            for i in range(0, len(texts), cfg.inference_batch_size):
                batch = tokenizer(list(texts[i:i + cfg.inference_batch_size]), truncation=True, max_length=cfg.max_length,
                                  padding=True, return_tensors="pt").to(best_model.device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    outputs.append(best_model(**batch).logits.argmax(-1).cpu())
        torch.cuda.synchronize()
        return torch.cat(outputs).numpy()

    return FitOutput(np.asarray(val_logits), np.asarray(test_logits), train_seconds, predict)


def assess(fit: FitOutput, val: pd.DataFrame, test: pd.DataFrame, cfg: ProjectConfig, classes: Sequence[str],
           classical_table: Optional[pd.DataFrame] = None, tfidf_baseline: Optional[float] = None,
           reference_predict: Optional[Callable[[pd.DataFrame], np.ndarray]] = None) -> Dict[str, Any]:
    """Turn logits into every Step 6 number: default and tuned-threshold scores, timing, drift slices, errors, verdict."""
    t = cfg.transformer
    y_val = transformer.label_ids(val[classical.LABEL], classes)
    y_test = transformer.label_ids(test[classical.LABEL], classes)
    p_val, p_test = transformer.positive_probability(fit.val_logits), transformer.positive_probability(fit.test_logits)
    if len(p_val) != len(y_val) or len(p_test) != len(y_test):
        raise ValueError("logits and labels have different lengths")
    threshold, val_f1 = transformer.best_threshold(y_val, p_val, t)
    default_pred, tuned_pred = transformer.apply_threshold(p_test, 0.5), transformer.apply_threshold(p_test, threshold)
    names = np.asarray(classes)
    default_eval = evaluate_classifier(names[y_test], names[default_pred], classes)
    tuned_eval = evaluate_classifier(names[y_test], names[tuned_pred], classes)
    texts = test["Text"].astype(str).head(t.inference_sample).tolist()
    predict_text = fit.predict
    predict_text(texts[:min(32, len(texts))])  # warm-up: the first GPU call is slow and would distort the timing
    timing = time_inference(predict_text, texts, t.inference_repeats)
    row = {"model": MODEL_NAME, "features": "raw text (subword tokens)", "best_param": f"threshold={threshold}",
           "val_macro_f1": round(val_f1, 4), "test_accuracy": round(tuned_eval["accuracy"], 4),
           "test_macro_f1": round(tuned_eval["macro_f1"], 4),
           **{f"f1_{c}": round(tuned_eval["per_class"][c]["f1"], 4) for c in classes},
           "tune_fit_seconds": round(fit.train_seconds, 1), "predict_seconds_per_1000": round(timing["seconds_per_1000"], 3)}
    results: Dict[str, Any] = {
        "model": t.model_name, "max_length": t.max_length, "epochs": t.epochs, "threshold": threshold,
        "default_threshold": {"accuracy": round(default_eval["accuracy"], 4), "macro_f1": round(default_eval["macro_f1"], 4)},
        "tuned_threshold": {"accuracy": round(tuned_eval["accuracy"], 4), "macro_f1": round(tuned_eval["macro_f1"], 4),
                            "per_class": tuned_eval["per_class"], "confusion_matrix": tuned_eval["confusion_matrix"]},
        "train_minutes": round(fit.train_seconds / 60, 1), "predict_seconds_per_1000": row["predict_seconds_per_1000"],
        "target_macro_f1": cfg.targets.macro_f1_binary,
        "meets_target": bool(tuned_eval["macro_f1"] > cfg.targets.macro_f1_binary),
    }
    if tfidf_baseline is not None:
        results["tfidf_baseline_macro_f1"] = tfidf_baseline
        results["gain_over_tfidf"] = round(tuned_eval["macro_f1"] - tfidf_baseline, 4)
        results["worth_the_cost"] = bool(worth_the_cost(tfidf_baseline, tuned_eval["macro_f1"], cfg.targets))
    if "review_date" in test.columns:
        predictions = {MODEL_NAME: tuned_pred}
        if reference_predict is not None:
            predictions["tfidf_best"] = label_array(reference_predict(test), classes)
        results["time_slices"] = transformer.slice_scores(test, predictions, y_test, t.slice_fraction)
    errors = transformer.confident_errors(test, p_test, tuned_pred, y_test, classes, t.error_rows)
    return {"results": results, "row": row, "table": transformer.combine_tables(classical_table, row),
            "errors": errors, "confusion_matrix": tuned_eval["confusion_matrix"]}


def label_array(labels: Sequence[Any], classes: Sequence[str]) -> np.ndarray:
    """Class names (or ids) from another model as 0/1 ids in the order of ``classes``."""
    values = np.asarray(labels)
    return values.astype(np.int64) if np.issubdtype(values.dtype, np.integer) else transformer.label_ids(list(values), classes)


def write_outputs(out: Dict[str, Any], cfg: ProjectConfig, classes: Sequence[str], sample: bool) -> Dict[str, Path]:
    """Save results JSON, the combined table, the errors and two charts; sample runs get a ``_sample`` suffix."""
    t, suffix = cfg.transformer, "_sample" if sample else ""
    cfg.paths.reports_path.mkdir(parents=True, exist_ok=True)
    paths = {"results": cfg.paths.reports_path / f"{t.results_name}{suffix}.json",
             "comparison": cfg.paths.reports_path / f"{t.comparison_name}{suffix}.csv",
             "errors": cfg.paths.reports_path / f"{t.errors_name}{suffix}.csv"}
    paths["results"].write_text(json.dumps({"sample": sample, **out["results"]}, indent=2), encoding="utf-8")
    out["table"].to_csv(paths["comparison"], index=False)
    out["errors"].to_csv(paths["errors"], index=False)
    figures = cfg.paths.figures_path
    plots.save_figure(plots.plot_confusion_matrix(out["confusion_matrix"], list(classes), f"{MODEL_NAME} (test set)"),
                      figures / f"confusion_matrix_transformer{suffix}.png", cfg.eda.dpi)
    plots.save_figure(plots.plot_model_comparison(out["table"]), figures / f"model_comparison_all{suffix}.png", cfg.eda.dpi)
    return paths


def _load_reference(cfg: ProjectConfig, sample: bool) -> Dict[str, Any]:
    """Full-run Step 5 outputs, if present: the comparison table, the TF-IDF baseline score and the best classical model."""
    ref: Dict[str, Any] = {"table": None, "baseline": None, "predict": None}
    if sample:  # the sample runs use different reviews, so comparing them with the full Step 5 run would be unfair
        return ref
    table_path = cfg.paths.reports_path / f"{cfg.classical.comparison_name}.csv"
    results_path = cfg.paths.reports_path / f"{cfg.classical.results_name}.json"
    model_path = cfg.paths.models_path / f"{cfg.classical.model_name}.joblib"
    if table_path.is_file():
        ref["table"] = pd.read_csv(table_path)
    if results_path.is_file():
        ref["baseline"] = json.loads(results_path.read_text(encoding="utf-8")).get("tfidf_baseline_macro_f1_for_step6")
    else:
        LOGGER.warning("%s not found; the verdict against TF-IDF will be skipped. Run Step 5 first.", results_path)
    if model_path.is_file():
        pipeline = joblib.load(model_path)
        ref["predict"] = lambda frame: pipeline.predict(frame[classical.TEXT])
    return ref


def run(cfg: ProjectConfig, sample: bool = False, fit_fn: Optional[FitFn] = None) -> Dict[str, Any]:
    """Split exactly as Step 5 did, fine-tune, assess, and write everything.

    Raises:
        FileNotFoundError: If Step 3 has not been run on the full data.
    """
    source = cfg.paths.text_path(sample=False)
    if not source.is_file():
        raise FileNotFoundError(f"{source} not found. Run `python -m reviews.preprocess` first.")
    frame = pd.read_parquet(source)
    if sample:  # same sampling call as Step 5's sample run, so the quick runs use the same reviews
        frame = frame.sample(n=min(cfg.transformer.sample_rows, len(frame)), random_state=classical.SEED)
    binary = classical.binary_frame(frame, cfg.labels)
    classes = list(cfg.labels.binary_classes)
    train, val, test = classical.stratified_split(binary, cfg.split)  # identical rule and seed to Step 5
    LOGGER.info("train %d, validation %d, test %d", len(train), len(val), len(test))
    model_dir = cfg.paths.models_path / (cfg.transformer.model_dir + ("_sample" if sample else ""))
    model_dir.mkdir(parents=True, exist_ok=True)
    fit = (fit_fn or fit_distilbert)(train, val, test, cfg.transformer, model_dir)
    ref = _load_reference(cfg, sample)
    out = assess(fit, val, test, cfg, classes, ref["table"], ref["baseline"], ref["predict"])
    (model_dir / "inference_config.json").write_text(json.dumps(
        {"threshold": out["results"]["threshold"], "max_length": cfg.transformer.max_length, "classes": classes}, indent=2),
        encoding="utf-8")
    paths = write_outputs(out, cfg, classes, sample)
    LOGGER.info("test macro-F1 %.4f (target %.2f); wrote %s", out["results"]["tuned_threshold"]["macro_f1"],
                cfg.targets.macro_f1_binary, paths["results"])
    return out["results"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT (Step 6).")
    parser.add_argument("--root", type=Path, default=Path("."), help="project root (default: current folder)")
    parser.add_argument("--sample", action="store_true", help="a 30,000-review smoke test, written to separate files")
    parser.add_argument("--epochs", type=int, default=None, help="training epochs (default 2)")
    parser.add_argument("--batch-size", type=int, default=None, help="training batch size (default 32; lower if the GPU runs out of memory)")
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING ...")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    base = ProjectConfig()
    overrides = {k: v for k, v in (("epochs", args.epochs), ("train_batch_size", args.batch_size)) if v is not None}
    cfg = replace(base, paths=replace(base.paths, root=args.root), transformer=replace(base.transformer, **overrides))
    run(cfg, sample=args.sample)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
