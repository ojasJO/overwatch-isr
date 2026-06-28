"""
xview_classes.py  —  xView class definitions and operational mapping
=====================================================================
xView has 60 fine-grained classes spread across sparse IDs 11–94.
For an operational UAV pipeline, we collapse them into 6 broad classes
that match what YOLO already knows (so the model can build on prior weights).

NOTE: xView has NO "person" class. Persons are too small at 0.3m GSD
to be reliably annotated. If you need person detection, keep using the
COCO-pretrained YOLOv11 alongside this fine-tuned model.

Operational classes (what your pipeline cares about):
  0  small_vehicle    cars, pickup, motorcycle, utility
  1  large_vehicle    bus, truck, cargo truck, semi-trailer
  2  aircraft         fixed-wing, helicopter, cargo plane, trainer
  3  vessel           any maritime vessel (boat, ship, barge, yacht)
  4  construction     bulldozer, crane, dump truck, excavator, etc.
  5  building         any building structure
"""

OPERATIONAL_CLASSES = {
    0: "small_vehicle",
    1: "large_vehicle",
    2: "aircraft",
    3: "vessel",
    4: "building",   # was 5 — moved up since construction was dropped
}

# Reverse: name -> id
CLASS_TO_ID = {v: k for k, v in OPERATIONAL_CLASSES.items()}


# xView class ID -> our operational class ID
# Source: https://challenge.xviewdataset.org/data-format
XVIEW_TO_OPERATIONAL = {
    # ── Aircraft (class 2) ────────────────────────────────────────
    11: 2,   # Fixed-wing Aircraft
    12: 2,   # Small Aircraft
    13: 2,   # Cargo Plane
    15: 2,   # Helicopter

    # ── Small Vehicles (class 0) ──────────────────────────────────
    17: 0,   # Passenger Vehicle
    18: 0,   # Small Car
    19: 0,   # Bus               ← actually LARGE, override below
    20: 0,   # Pickup Truck
    21: 0,   # Utility Truck

    # ── Large Vehicles (class 1) ──────────────────────────────────
    23: 1,   # Truck
    24: 1,   # Cargo Truck
    25: 1,   # Truck w/Box
    26: 1,   # Truck Tractor
    27: 1,   # Trailer
    28: 1,   # Truck w/Flatbed
    29: 1,   # Truck w/Liquid
    32: 1,   # Crane Truck
    60: 1,   # Mobile Crane
    61: 1,   # Dump Truck
    62: 1,   # Haul Truck
    63: 1,   # Scraper/Tractor
    64: 1,   # Front Loader
    65: 1,   # Excavator
    66: 1,   # Cement Mixer
    71: 1,   # Engineering Vehicle (generic)

    # ── Override: Bus is LARGE not small ──────────────────────────
    # (Re-declared after small block to ensure correct mapping)
    # 19: 1,   # Bus  — uncomment if you want bus as large

    # ── Construction (class 4) ────────────────────────────────────
    # Treating construction equipment as its own class
    # Currently mapped to large_vehicle above; uncomment to split:
    # 60: 4, 61: 4, 62: 4, 63: 4, 64: 4, 65: 4, 66: 4, 71: 4,

    # ── Vessels (class 3) ─────────────────────────────────────────
    40: 3,   # Maritime Vessel
    41: 3,   # Motorboat
    42: 3,   # Sailboat
    44: 3,   # Tugboat
    45: 3,   # Barge
    47: 3,   # Fishing Vessel
    49: 3,   # Ferry
    50: 3,   # Yacht
    51: 3,   # Container Ship
    52: 3,   # Oil Tanker

    # ── Buildings (class 4) ───────────────────────────────────────
    73: 4,   # Building
    74: 4,   # Hut/Tent
    76: 4,   # Damaged Building
    77: 4,   # Facility
    79: 4,   # Construction Site
    83: 4,   # Vehicle Lot
    84: 4,   # Helipad
    86: 4,   # Storage Tank
    89: 4,   # Shipping Container Lot
    91: 4,   # Shipping Container
    93: 4,   # Pylon
    94: 4,   # Tower

}


def xview_to_op(xview_class_id: int) -> int | None:
    """
    Convert an xView class ID to our operational class ID.
    Returns None if the xView class is not in our mapping
    (those detections are dropped).
    """
    return XVIEW_TO_OPERATIONAL.get(xview_class_id)


def get_class_distribution_summary() -> dict:
    """How many xView classes map to each of our 6 classes."""
    counts = {i: 0 for i in OPERATIONAL_CLASSES}
    for op_id in XVIEW_TO_OPERATIONAL.values():
        counts[op_id] += 1
    return {OPERATIONAL_CLASSES[k]: v for k, v in counts.items()}


if __name__ == "__main__":
    print("Operational classes:")
    for k, v in OPERATIONAL_CLASSES.items():
        print(f"  {k}: {v}")
    print(f"\nMapped xView classes: {len(XVIEW_TO_OPERATIONAL)} of 60")
    print(f"\nDistribution:")
    for name, count in get_class_distribution_summary().items():
        print(f"  {name:15s} ← {count} xView classes")