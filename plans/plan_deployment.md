# Prototype Deployment on Jetson Orin Nano

## Goal

Build a **non-actuating prototype deployment** of the self-regulating AV stack on a Jetson Orin Nano. The prototype demonstrates that the simulation-trained actor policy can run on real edge hardware using camera/GPS-derived observations and can produce real-time advisory feedback on a display.

The prototype is not used for vehicle control. It only displays recommended behavior such as:

```text
Recommended speed: 42 mph
Lane suggestion: keep lane / prepare left / prepare right
Traffic state: high density ahead
Confidence: medium
```

This should be framed as a **hardware-in-the-loop / edge deployment demonstration**, not a full autonomous driving system.

---

## Deployment objective

The prototype should answer four questions:

1. Can Jetson Orin Nano run the perception stack in real time?
2. Can the perception stack produce the same observation variables used by the simulated AV actor?
3. Can the simulation-trained actor policy run on-device with low latency?
4. Can the system produce interpretable driver-facing feedback through a display?

The demo should support the paper’s claim that self-regulating AV policies can be implemented as **vehicle-edge intelligence**, even before any closed-loop actuation.

---

## Hardware setup

Use the following hardware components:

```text
Jetson Orin Nano
USB or CSI camera mounted near windshield
GPS module connected over USB/serial
Optional IMU
Optional OBD-II adapter for vehicle speed, if available
Portable touchscreen or HDMI display
Power supply / car power adapter
Local SSD or SD card for logging
```

Minimum viable setup:

```text
Jetson Orin Nano + camera + GPS + display
```

Preferred setup:

```text
Jetson Orin Nano + camera + GPS + OBD-II speed + display + local logging
```

The OBD-II adapter is optional but useful because GPS speed can be noisy at low speed or under poor reception.

---

## Software modules

Add a new deployment folder:

```text
deployment/
  jetson/
    README.md
    requirements_jetson.txt
    run_demo.py
    config.yaml

    sensors/
      camera_stream.py
      gps_reader.py
      obd_reader.py
      time_sync.py

    perception/
      vehicle_detector.py
      tracker.py
      depth_estimator.py
      observation_builder.py

    policy/
      actor_model.py
      export_policy.py
      policy_inference.py

    ui/
      display_app.py
      dashboard.py

    logging/
      logger.py
      video_logger.py
      metadata_logger.py

    calibration/
      camera_calibration.py
      gps_alignment.py
```

---

## Perception and sensing pipeline

The Jetson prototype should convert raw sensor inputs into the same observation vector used by the actor in simulation.

Pipeline:

```text
Camera frame
  -> vehicle detection
  -> tracking
  -> depth / distance estimation
  -> local traffic feature extraction
  -> observation vector
  -> actor policy
  -> recommended speed/lane feedback
  -> display + log
```

The observation builder should produce variables such as:

```text
ego speed
ego GPS location
ego heading
current road segment if map-matched
nearby vehicle count
estimated local density
leader distance
leader relative speed
left/right lane occupancy estimate, if available
density bins by distance
mean relative speed of nearby vehicles
distance to known merge/bottleneck, if map data is provided
```

Initial version can be simpler:

```text
ego speed
nearby vehicle count
front vehicle distance
front vehicle relative speed
coarse density estimate
GPS position
heading
```

Then later add lane-level features.

---

## Computer vision models

Use lightweight edge models first.

Suggested stack:

```text
Vehicle detection: YOLOv8n / YOLOv8s
Tracking: ByteTrack or simple SORT-style tracker
Depth: Depth Anything small / lightweight monocular depth model
Policy: small MLP actor exported from PyTorch
```

The perception stack should output metadata, not raw video, for the control policy:

```json
{
  "timestamp": 123.45,
  "ego_speed": 18.2,
  "gps_lat": 40.123,
  "gps_lon": -74.123,
  "num_detected_vehicles": 7,
  "leader_distance": 24.5,
  "leader_relative_speed": -2.1,
  "density_0_20m": 0.05,
  "density_20_50m": 0.12,
  "density_50_100m": 0.08
}
```

This metadata should match the simulation observation format as closely as possible.

---

## Actor policy deployment

The trained simulation actor should be exported from PyTorch.

Recommended first version:

```text
PyTorch checkpoint -> TorchScript model -> Jetson inference
```

Later optional versions:

```text
ONNX export
TensorRT optimization
FP16 inference
INT8 quantization
```

Actor interface:

```python
obs = observation_builder.build(...)
action = actor(obs)

desired_speed = action["desired_speed"]
desired_lane = action["desired_lane"]
confidence = action["confidence"]
```

The deployed actor should not directly command the vehicle. It only produces advisory output.

---

## Display / dashboard

The dashboard should be simple and readable.

Display fields:

```text
Current speed
Recommended speed
Recommended lane behavior
Detected traffic density
Number of nearby vehicles
System confidence
Perception FPS
Policy latency
GPS status
Logging status
```

Example display:

```text
Current speed:      46 mph
Recommended speed: 41 mph
Lane advice:       Keep lane
Traffic density:   High
Detected vehicles: 8
Confidence:        Medium
FPS:               12.4
Latency:           86 ms
```

Important: the display should be positioned for demo visibility, not for actual driver use during normal driving. The feedback should not be followed while driving.

---

## Safety and demo constraints

The prototype must be explicitly non-actuating.

Rules:

```text
No connection to throttle, brake, steering, or cruise control.
No driver should follow the recommendation during the demo drive.
The display is for logging/demo purposes only.
Run with a passenger/operator when collecting data.
Prefer controlled roads, parking lots, or low-risk data-collection settings.
Store logs for offline analysis.
```

Phrase this in the paper as:

> We deploy the perception and policy-inference stack on Jetson Orin Nano as an advisory-only prototype. The device produces real-time recommendations but is not connected to vehicle actuation and is not used for driving decisions.

---

## Prototype evaluation metrics

Measure edge performance separately from traffic-control performance.

### Perception metrics

```text
vehicle detection FPS
tracking stability
number of detected vehicles
distance estimate consistency
detection range
dropped-frame rate
```

### Policy inference metrics

```text
actor inference latency
end-to-end latency from frame capture to recommendation
CPU/GPU utilization
memory usage
power draw if available
```

### Observation quality metrics

```text
density estimate over time
leader distance estimate over time
relative speed estimate over time
GPS speed vs OBD/GPS comparison
observation missingness
```

### System metrics

```text
full pipeline FPS
average latency
95th percentile latency
metadata size per second
log size per minute
runtime stability
```

A useful target:

```text
perception + observation + actor + display latency < 200 ms
```

---

## Simulation-to-prototype alignment

A key goal is to show that the simulation actor receives a compatible observation vector in deployment.

Add a mapping table:

| Simulation observation | Jetson prototype source                          |
| ---------------------- | ------------------------------------------------ |
| Ego speed              | GPS / OBD-II                                     |
| Ego segment            | GPS map matching                                 |
| Leader gap             | Camera + depth                                   |
| Relative speed         | Tracker + depth over time                        |
| Local density          | Detected vehicles within distance bins           |
| Lane occupancy         | Camera detections + lane geometry, later version |
| Distance to merge      | GPS + preloaded map                              |
| Sensing noise          | Real perception uncertainty                      |

This table will make the deployment story much clearer.

---

## Development tasks

### Task P1: Jetson setup

```text
Install JetPack-compatible Python environment.
Install PyTorch for Jetson.
Install OpenCV, NumPy, PyYAML, camera drivers.
Verify camera stream.
Verify GPS stream.
Verify display output.
```

Deliverable:

```text
run_demo.py opens camera, reads GPS, displays live status.
```

---

### Task P2: Vehicle detection and tracking

```text
Run YOLO vehicle detector on camera frames.
Add tracker for stable vehicle IDs.
Log bounding boxes, IDs, confidence, FPS.
```

Deliverable:

```text
demo video with detected/tracked vehicles.
```

---

### Task P3: Depth and distance estimation

```text
Run lightweight monocular depth model.
Estimate approximate distance to detected vehicles.
Smooth estimates over time.
Compute leader distance and density bins.
```

Deliverable:

```text
metadata log with vehicle distances over time.
```

---

### Task P4: Observation builder

```text
Convert perception + GPS into actor observation vector.
Normalize using simulation training statistics.
Handle missing values.
Add confidence flags.
```

Deliverable:

```text
real_observation.npy or real_observation.jsonl logs compatible with actor input.
```

---

### Task P5: Actor export and inference

```text
Export trained actor from simulation.
Load actor on Jetson.
Run actor on live observations.
Measure latency.
```

Deliverable:

```text
policy_inference.py outputs desired speed/lane recommendations in real time.
```

---

### Task P6: Dashboard

```text
Build simple display interface.
Show current speed, recommended speed, lane advice, traffic density, FPS, latency.
```

Deliverable:

```text
Jetson display demo running in real time.
```

---

### Task P7: Offline replay mode

This is important for debugging.

```text
Record camera/GPS logs.
Replay logs through perception/observation/policy pipeline.
Compare live vs replay outputs.
```

Deliverable:

```text
python deployment/jetson/replay_demo.py --log <log_dir>
```

---

### Task P8: Prototype demo

```text
Mount camera and GPS.
Run advisory-only system in vehicle.
Record dashboard and logs.
Do not use recommendations for driving.
Analyze latency and observation quality afterward.
```

Deliverable:

```text
prototype deployment figure/table:
  FPS
  latency
  detected vehicle count
  policy inference time
  example dashboard screenshot
```

---

## Paper positioning

In the paper, this should be a short but valuable section:

**Prototype Deployment on Edge Hardware**

Main message:

> To evaluate deployability, we implement the sensing and policy-inference stack on a Jetson Orin Nano. A windshield-mounted camera and GPS module provide local observations, an edge perception pipeline estimates nearby traffic state, and the simulation-trained actor produces advisory speed/lane recommendations on a display. The prototype is non-actuating and is used only to validate real-time feasibility of local sensing and policy inference.

This section should support the claim that the method is not merely a simulator policy; it has a plausible path to edge deployment.
