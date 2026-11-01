# Synthetic camera result contract

Topic: `synthetic/camera-01/inspection/v2`; schema version: integer `2`.
`camera_quality/messages.py` is the executable schema. Payloads are limited to
64 kB, unknown fields are rejected, and dimensions are bounded to 8–4096 pixels.
The simulator currently generates only 96×96 frames.

```json
{
  "schema_version": 2,
  "image_id": "sim-aaaaaaaaaaaaaaaaaaaaaaaa",
  "camera_id": "synthetic-camera-01",
  "source": "synthetic",
  "captured_at": "2026-01-01T00:00:00+00:00",
  "width": 96,
  "height": 96,
  "opencv": {"decision": "pass", "regions": []},
  "cnn": {"decision": "pass", "probability": 0.2, "threshold": 0.5},
  "yolo": {
    "status": "not_run",
    "model": null,
    "initialization": "not_run",
    "weights_sha256": null,
    "confidence_threshold": 0.25,
    "decision": "not_run",
    "boxes": []
  },
  "decision": "pass"
}
```

This is a schema example, not a measured model prediction. Each OpenCV region
contains integer `x`, `y`, `width`, `height`, positive `area_px`, and floating-point
`center_x`, `center_y`. Boxes/centroids must be inside the image. Areas and positions
are pixel measurements, not millimeters. CNN probabilities are uncalibrated scores.
The combined decision is an OR across evaluated methods; `not_run` is not a pass.
The evaluation compares individual methods, not the combined decision.

An evaluated YOLO result has `status="evaluated"`, `model="yolov8n"`, initialization
`coco_pretrained` or `scratch`, and the SHA-256 of its evaluated checkpoint.
Its `boxes` are floating-point pixel x/y/width/height plus integer class_id (0–2),
class_name (`scratch`, `dent`, `hole`), and confidence (finite 0–1, at least the
published confidence_threshold). IDs/names must agree, boxes must stay inside the
image, and a nonempty boxes list requires decision `fail`. An evaluated empty list
is `pass`; unavailable/unevaluated inference must use `not_run` with no model/weights.
At most 100 YOLO boxes are accepted per message. Ground-truth boxes are never sent.

Image IDs are the first 24 hex characters of SHA-256 of grayscale pixels, prefixed
`sim-`. The consumer validates the ID format; images are not transported, so it
cannot independently verify pixel integrity. Hash collisions are not mathematically
impossible. Fixed simulated timestamps and full payloads make exact replay stable.

SQLite table `inspections_v2` stores explicit ID/time/source/decision/probability columns, the validated
canonical JSON, and its full SHA-256. Parameterized SQL and a primary key enforce
idempotency, including concurrent duplicate delivery. Different canonical content
with an existing ID is rejected instead of silently overwriting results. A new
model's results for the same image therefore require a new database/schema design,
not an overwrite hidden as replay. This example is a first-result store, not a
model-version history system.

Version 2 also stores yolo_status, yolo_decision and yolo_boxes_json as explicit
columns; the full detector provenance remains in canonical payload_json. A v1
payload is rejected rather than silently treated as a detector pass. Existing
v1 `inspections` tables are not modified; v2 loads use a separate table. This
deliberate schema boundary avoids silently overwriting old first-result records.

Generator bounding boxes are simulation labels. For soft Gaussian dents the box
encloses pixels where the dent mask exceeds 15% of its peak, rather than a physical
edge. Classical contour area and the simulated label box need not coincide.

The mock synchronously calls the real validation/load function. Tests also mock
paho callbacks to check publish options, acknowledgement ordering, invalid
messages, and consumer replay. They do **not** exercise a broker or sockets.
The optional local paho consumer commits valid results before acknowledgement,
logs/discards invalid/conflicting messages, and disconnects without ack on SQLite
failure. There is no dead-letter queue, automatic recovery, or durable publisher
outbox. No cloud persistence or remote service exists in this repository.
