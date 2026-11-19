# Model inference fixture attribution

`pothole_test.jpg` and `pothole_test.txt` are an image/label pair from HRP4K test image `30091.jpg`, retrieved through the unofficial `dronefreak/HRP4K` mirror. The source, author attribution, release versions, and CC BY 4.0 license trail are documented in [`tests/evaluation/hrp4k/README.md`](../evaluation/hrp4k/README.md).

This fixture exercises real inference and annotation output. It is not a successful-detection accuracy assertion: on this frame the model's candidate box lands on a dark pavement shadow and does not overlap the labeled pothole at IoU 0.50. See the measured sample report under `data/model_eval/`.
