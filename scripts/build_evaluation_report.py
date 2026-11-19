"""Build the visual, evidence-linked RoadGuard model evaluation HTML report."""
import csv
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'model_eval'


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    summary = load_json(OUT / 'hrp4k_sample_summary.json')
    with (OUT / 'hrp4k_sample_results.csv').open(encoding='utf-8', newline='') as stream:
        records = list(csv.DictReader(stream))
    scores = [float(score) for row in records for score in row['confidences'].split(';') if score]
    tp = summary['true_positive_boxes']
    fp = summary['false_positive_boxes']
    fn = summary['missed_ground_truth_boxes']
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    positives = [row for row in records if int(row['false_positive_boxes_iou50']) > 0]
    misses = sorted((row for row in records if int(row['missed_ground_truth_boxes_iou50']) > 0),
                    key=lambda row: int(row['missed_ground_truth_boxes_iou50']), reverse=True)

    def esc(value):
        return html.escape(str(value))

    def metric(value):
        return 'N/A (no predictions)' if value is None else f'{value:.3f} ({value * 100:.1f}%)'

    rows = ''.join(
        '<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(
            esc(r['image']), int(r['ground_truth']), int(r['predictions']),
            int(r['true_positive_boxes_iou50']), int(r['false_positive_boxes_iou50']),
            int(r['missed_ground_truth_boxes_iou50']), esc(r['confidences'] or '—'))
        for r in records
    )
    confidence_text = (f'{len(scores)} predicted box: ' + ', '.join(f'{s:.4f}' for s in scores)
                       if scores else 'No predicted boxes.')
    hist = []
    for low in range(0, 100, 10):
        count = sum(low / 100 <= score < (low + 10) / 100 for score in scores)
        height = max(2, round(90 * count / max(1, len(scores))))
        hist.append(f'<div class="bar-wrap"><span>{count}</span><div class="bar" style="height:{height}px"></div><small>{low/100:.1f}–{(low+10)/100:.1f}</small></div>')

    # Use compact local image references; annotated examples remain separately inspectable.
    sample_cards = [
        ('30091', 'False positive + missed pothole', 'Cyan is the model box (0.4689); green is ground truth. The cyan box lands on a pavement shadow.', 'FP'),
        ('30811', 'Three missed potholes', 'Ground-truth boxes are green; the model returned no boxes.', 'FN'),
        ('30630', 'Two missed potholes', 'Both labeled boxes were missed at confidence 0.25.', 'FN'),
        ('30090', 'Clean-road negative', 'No ground-truth pothole and no model prediction.', 'TN image'),
    ]
    cards = ''.join(
        f'<article class="sample"><img src="{image_id}_evaluation.jpg" alt="HRP4K {image_id} annotated road image">'
        f'<div><span class="tag">{tag}</span><h3>{title}</h3><p>{desc}</p>'
        f'<a href="{image_id}_evaluation.jpg">Open full-size image</a></div></article>'
        for image_id, title, desc, tag in sample_cards
    )

    fp_rows = ''.join(f'<li><b>{esc(row["image"])}</b>: {row["false_positive_boxes_iou50"]} false-positive box</li>' for row in positives)
    if not fp_rows:
        fp_rows = '<li>No false-positive boxes.</li>'
    missed_rows = ''.join(f'<li><b>{esc(row["image"])}</b>: {row["missed_ground_truth_boxes_iou50"]} missed box(es)</li>' for row in misses)

    sweep = []
    for name, conf, iou in [('conf050', 0.50, 0.45), ('iou030', 0.25, 0.30), ('iou065', 0.25, 0.65)]:
        data = load_json(OUT / name / 'hrp4k_sample_summary.json')
        p = data['true_positive_boxes'] / (data['true_positive_boxes'] + data['false_positive_boxes']) if data['predicted_boxes'] else None
        r = data['true_positive_boxes'] / data['ground_truth_boxes'] if data['ground_truth_boxes'] else None
        f = 2 * data['true_positive_boxes'] / (2 * data['true_positive_boxes'] + data['false_positive_boxes'] + data['missed_ground_truth_boxes']) if 2 * data['true_positive_boxes'] + data['false_positive_boxes'] + data['missed_ground_truth_boxes'] else None
        sweep.append(f'<tr><td>{conf:.2f}</td><td>{iou:.2f}</td><td>{data["predicted_boxes"]}</td><td>{data["true_positive_boxes"]}</td><td>{data["false_positive_boxes"]}</td><td>{data["missed_ground_truth_boxes"]}</td><td>{metric(p)}</td><td>{metric(r)}</td><td>{metric(f)}</td></tr>')
    baseline_sweep = f'<tr><td>0.25</td><td>0.45</td><td>{summary["predicted_boxes"]}</td><td>{tp}</td><td>{fp}</td><td>{fn}</td><td>{metric(precision)}</td><td>{metric(recall)}</td><td>{metric(f1)}</td></tr>'

    report = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RoadGuard AI Model Evaluation</title>
<style>
:root{{--navy:#081525;--panel:#0f1e30;--border:#26384d;--text:#f8fafc;--muted:#cbd5e1;--cyan:#22d3ee;--green:#22c55e;--amber:#fbbf24;--red:#fb7185}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--navy);color:var(--text);font:16px/1.55 Segoe UI,Arial,sans-serif}}main{{max-width:1180px;margin:auto;padding:36px 22px 64px}}h1{{font-size:clamp(2rem,4vw,3.1rem);line-height:1.1;margin:.4rem 0}}h2{{margin:2.6rem 0 1rem;font-size:1.55rem}}h3{{margin:.2rem 0 .4rem}}p,li{{color:var(--muted)}}a{{color:var(--cyan)}}.eyebrow,.tag{{color:var(--cyan);font-size:.78rem;letter-spacing:.09em;text-transform:uppercase;font-weight:700}}.lead{{max-width:860px;color:var(--muted);font-size:1.1rem}}.notice{{border-left:4px solid var(--amber);padding:14px 18px;background:#2a2619;border-radius:8px;margin:20px 0}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:24px 0}}.stat,.panel,.sample{{background:var(--panel);border:1px solid var(--border);border-radius:12px}}.stat{{padding:16px}}.stat small{{display:block;color:var(--muted)}}.stat strong{{display:block;font-size:1.75rem;margin-top:4px}}.panel{{padding:18px;margin:14px 0}}.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;min-width:740px}}th,td{{text-align:left;padding:10px 12px;border-bottom:1px solid var(--border);vertical-align:top}}th{{color:var(--cyan);font-size:.86rem}}.bars{{height:140px;display:flex;align-items:end;gap:8px;padding:8px 0 0}}.bar-wrap{{flex:1;text-align:center;color:var(--muted);font-size:.75rem}}.bar{{background:var(--cyan);min-height:2px;border-radius:5px 5px 0 0;margin:4px auto;width:60%;max-width:42px}}.samples{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}}.sample{{overflow:hidden}}.sample img{{width:100%;height:auto;display:block;background:#000}}.sample div{{padding:14px 16px}}.sample p{{margin:.25rem 0 .6rem}}.tag{{background:#102d3a;border-radius:20px;padding:3px 9px;letter-spacing:.03em}}.key{{display:flex;gap:18px;color:var(--muted);font-size:.9rem}}.dot{{display:inline-block;width:12px;height:12px;border-radius:50%;background:var(--green);margin-right:6px}}.dot.cyan{{background:var(--cyan)}}footer{{border-top:1px solid var(--border);margin-top:40px;padding-top:18px;color:var(--muted);font-size:.9rem}}
@media(max-width:600px){{main{{padding:24px 14px}}.samples{{grid-template-columns:1fr}}}}
</style></head><body><main>
<div class="eyebrow">RoadGuard AI · Checkpoint evaluation</div><h1>Small held-out test, weak detection</h1>
<p class="lead">A box-level evaluation of the pinned YOLOv8s checkpoint on ten labeled HRP4K test images, including clean-road negatives. These results are a diagnostic sample and are not a production accuracy estimate.</p>
<div class="notice"><b>Observed result:</b> at confidence 0.25 and NMS IoU 0.45, the model produced one box. It matched none of the eight labeled pothole boxes: 1 false-positive box and 8 missed boxes. The candidate is visibly on a pavement shadow.</div>
<div class="grid">
<div class="stat"><small>Test images</small><strong>{summary['images']}</strong></div><div class="stat"><small>With potholes</small><strong>{summary['positive_images']}</strong></div><div class="stat"><small>Without potholes</small><strong>{summary['negative_images']}</strong></div><div class="stat"><small>Ground-truth boxes</small><strong>{summary['ground_truth_boxes']}</strong></div><div class="stat"><small>True-positive boxes</small><strong>{tp}</strong></div><div class="stat"><small>False-positive boxes</small><strong>{fp}</strong></div><div class="stat"><small>Missed boxes</small><strong>{fn}</strong></div></div>
<section><h2>Metrics</h2><div class="panel"><p>Valid YOLO label files exist for all ten images. Matching is one-to-one with a predicted/ground-truth box IoU threshold of 0.50.</p><div class="table-wrap"><table><thead><tr><th>Precision</th><th>Recall</th><th>F1</th><th>Basis</th></tr></thead><tbody><tr><td>{metric(precision)}</td><td>{metric(recall)}</td><td>{metric(f1)}</td><td>TP={tp}, FP={fp}, FN={fn}, box-level</td></tr></tbody></table></div><p>Precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = 2TP/(2TP+FP+FN). The values describe only this 10-image sample and this operating threshold.</p></div></section>
<section><h2>Confidence distribution</h2><div class="panel"><p>{esc(confidence_text)} Nine images had no prediction, so there are no confidence scores for those images.</p><div class="bars">{''.join(hist)}</div><div class="key"><span>Prediction confidence bin</span><span>One score at 0.4689</span></div></div></section>
<section><h2>False positives and misses by image</h2><div class="panel"><h3>Most false positives</h3><ul>{fp_rows}</ul><h3>Missed ground-truth boxes</h3><ul>{missed_rows}</ul></div><div class="table-wrap"><table><thead><tr><th>Image</th><th>GT boxes</th><th>Predictions</th><th>TP</th><th>FP</th><th>FN</th><th>Confidence(s)</th></tr></thead><tbody>{rows}</tbody></table></div></section>
<section><h2>Annotated examples</h2><p class="key"><span><i class="dot"></i>HRP4K ground truth</span><span><i class="dot cyan"></i>Model prediction</span></p><div class="samples">{cards}</div></section>
<section><h2>Threshold check and recommendation</h2><div class="panel"><div class="table-wrap"><table><thead><tr><th>Confidence</th><th>NMS IoU</th><th>Pred.</th><th>TP</th><th>FP</th><th>FN</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>{baseline_sweep}{''.join(sweep)}</tbody></table></div><p>Raising confidence from 0.25 to 0.50 removes the sole 0.4689 false-positive candidate but still misses all eight labeled potholes. Changing NMS IoU from 0.30 to 0.65 leaves these counts unchanged. Keep confidence at 0.25 and NMS IoU at 0.45 for diagnostic comparison; for any operational trial, use at least 0.50 in review-only mode, but do not treat it as a reliable detector setting. This sample cannot identify a useful accuracy/recall trade-off. IoU is not tunable from this result because each image produced at most one prediction.</p></div></section>
<section><h2>Limitations</h2><div class="panel"><ul><li>Only ten test images were scored, with eight labeled potholes; the sample is too small for a deployment claim.</li><li>The checkpoint found no correctly localized boxes and missed all eight labels at confidence 0.25.</li><li>Only one confidence score was observed, so confidence calibration and score distributions are not meaningfully assessed.</li><li>Images come from one benchmark split and may not represent RoadGuard’s deployment geography, camera angle, pavement, weather, or video motion.</li><li>High-resolution road details can become very small at the checkpoint’s 640-pixel inference size; this may contribute to missed small defects, but this sample cannot establish the cause.</li><li>The evaluation is still-image box matching, not temporal video tracking or a frame-level video benchmark.</li></ul></div></section>
<section><h2>Checkpoint and licensing provenance</h2><div class="panel"><p><b>Checkpoint:</b> {esc(Path(summary['model']).name)}, {summary['model_size_bytes']:,} bytes; SHA-256 <code>{esc(summary['model_sha256'])}</code>. Inference ran on Ultralytics 8.4.215 / PyTorch 2.15.0+cpu. The <a href="https://huggingface.co/peterhdd/pothole-detection-yolov8">model card</a> labels the checkpoint Apache-2.0, describes YOLOv8s / one pothole class / 22.5 MB, and links its training data to <code>Ryukijano/Pothole-detection-Yolov8</code>.</p><p><b>Training-data discrepancy:</b> the linked Hugging Face dataset metadata says OpenRAIL, while its embedded dataset YAML refers to the Roboflow project and the <a href="https://universe.roboflow.com/project-ssayl/potholes-detection-d4rma/dataset/1">upstream Roboflow page</a> lists CC BY 4.0. Those records conflict. The model-card license does not by itself resolve the training-data terms; do not infer permission for model redistribution or commercial deployment from it. This report’s evaluation sample is separate from that source.</p><p><b>Evaluation data:</b> the images came from the unofficial <code>dronefreak/HRP4K</code> v1.00 mirror, which states CC BY 4.0 and identifies the original Zenodo release. The NLM catalog lists the complete v1.01 Zenodo dataset as CC BY 4.0. The dataset record’s license is separate from the linked paper’s CC BY-NC-ND article license. See the <a href="https://huggingface.co/datasets/dronefreak/HRP4K/blob/main/README.md">mirror attribution and split note</a>, <a href="https://datasetcatalog.nlm.nih.gov/dataset?q=0003245906">NLM/Zenodo record</a>, and <a href="https://doi.org/10.1038/s41597-026-07317-w">HRP4K paper</a>.</p></div></section>
<section><h2>Reproducibility</h2><div class="panel"><p>Thresholds: confidence {summary['confidence_threshold']:.2f}, NMS IoU {summary['nms_iou_threshold']:.2f}; matching IoU 0.50. Model SHA-256 is recorded above and in the summary JSON. Per-image coordinates and outcome labels are in <a href="hrp4k_sample_results.csv">the CSV</a>; the aggregate is in <a href="hrp4k_sample_summary.json">the JSON summary</a>.</p><p>Sample IDs: 30090, 30091, 30270, 30269, 30455, 30450, 30632, 30630, 30810, 30811.</p></div></section>
<footer>Generated from the saved RoadGuard evaluation artifacts. Green boxes show labeled ground truth; cyan boxes show model outputs.</footer>
</main></body></html>'''
    path = OUT / 'roadguard_model_evaluation_report.html'
    path.write_text(report, encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
