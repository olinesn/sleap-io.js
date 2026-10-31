"""Print what Python sleap-io reads from annotated .slp files, as JSON.

For tests/python-compat-annotations.test.ts, and to try a saved file on another
version by hand:

    uv run --no-project --with "sleap-io==0.9.5" --with shapely \\
        python -I tests/python-read-annotations.py file.slp

Each path gets the annotations read from it eagerly and lazily, and the path of a
copy Python saved (beside it, `<path>.resaved.slp`), or the error that stopped each.
"""

import json
import math
import sys
import warnings

import shapely
import sleap_io as sio


def number(value):
    """Return a float, with NaN as None so that the output is valid JSON."""
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value


def embedding(value):
    """Return an embedding's values, as a plain list."""
    return None if value is None else [float(v) for v in value.vector]


def category(annotation):
    """Return a category's name, however this version of sleap-io holds it."""
    value = annotation.category
    return getattr(value, "name", value) or ""


def common(annotation, labels, instances):
    """Return the fields every annotation kind shares."""
    instance = annotation.instance
    return {
        "name": annotation.name,
        "category": category(annotation),
        "source": annotation.source,
        "track": None if annotation.track is None else annotation.track.name,
        "tracking_score": number(annotation.tracking_score),
        "instance": None if instance is None else instances[id(instance)],
        "identity": None if annotation.identity is None else annotation.identity.name,
        "identity_score": number(annotation.identity_score),
        "category_score": number(annotation.category_score),
        "identity_embedding": embedding(annotation.identity_embedding),
        "category_embedding": embedding(annotation.category_embedding),
    }


def roi_entry(roi, labels, instances):
    geometry = roi.geometry
    n_points = len(shapely.get_coordinates(geometry))
    return {
        **common(roi, labels, instances),
        "type": type(roi).__name__,
        "wkt": geometry.wkt if n_points < 100 else f"{geometry.geom_type}: {n_points}",
        "score": number(getattr(roi, "score", None)),
    }


def mask_entry(mask, labels, instances, masks):
    score_map = getattr(mask, "score_map", None)
    source = getattr(mask, "from_predicted", None)
    return {
        **common(mask, labels, instances),
        "type": type(mask).__name__,
        "rle": mask.rle_counts.tolist(),
        "height": mask.height,
        "width": mask.width,
        "scale": [number(v) for v in mask.scale],
        "offset": [number(v) for v in mask.offset],
        "score": number(getattr(mask, "score", None)),
        "score_map": None if score_map is None else score_map.tolist(),
        "score_map_scale": [number(v) for v in getattr(mask, "score_map_scale", [])],
        "score_map_offset": [number(v) for v in getattr(mask, "score_map_offset", [])],
        "from_predicted": None if source is None else masks[id(source)],
    }


def label_image_entry(li, labels, instances):
    score_map = getattr(li, "score_map", None)
    objects = {}
    for label_id, info in li.objects.items():
        objects[str(label_id)] = {
            "name": info.name,
            "category": getattr(info.category, "name", info.category) or "",
            "track": None if info.track is None else info.track.name,
            "score": number(info.score),
            "tracking_score": number(info.tracking_score),
            "instance": None if info.instance is None else instances[id(info.instance)],
        }
    return {
        "type": type(li).__name__,
        "data": li.data.tolist(),
        "height": li.height,
        "width": li.width,
        "source": li.source,
        "objects": objects,
        "scale": [number(v) for v in li.scale],
        "offset": [number(v) for v in li.offset],
        "score": number(getattr(li, "score", None)),
        "score_map": None if score_map is None else score_map.tolist(),
        "score_map_scale": [number(v) for v in getattr(li, "score_map_scale", [])],
        "score_map_offset": [number(v) for v in getattr(li, "score_map_offset", [])],
    }


def bbox_entry(bbox, labels, instances):
    return {
        **common(bbox, labels, instances),
        "type": type(bbox).__name__,
        "xyxy": [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
        "angle": bbox.angle,
        "score": number(getattr(bbox, "score", None)),
    }


def centroid_entry(centroid, labels, instances):
    return {
        **common(centroid, labels, instances),
        "type": type(centroid).__name__,
        "xy": [centroid.x, centroid.y],
    }


def read(labels):
    """Return every annotation with where Python put it: video and frame number."""
    # An instance by its position in the file, to check that links land on the right
    # one rather than merely on some instance.
    instances = {}
    for lf in labels.labeled_frames:
        for instance in lf.instances:
            instances[id(instance)] = len(instances)
    masks = {}
    for lf in labels.labeled_frames:
        for mask in lf.masks:
            masks[id(mask)] = len(masks)

    out = {
        "n_instances": len(instances),
        "rois": [],
        "masks": [],
        "label_images": [],
        "bboxes": [],
        "centroids": [],
        "static_rois": [],
    }
    for lf in sorted(
        labels.labeled_frames,
        key=lambda lf: (labels.videos.index(lf.video), lf.frame_idx),
    ):
        where = {"video": labels.videos.index(lf.video), "frame_idx": int(lf.frame_idx)}
        for roi in lf.rois:
            out["rois"].append({**where, **roi_entry(roi, labels, instances)})
        for mask in lf.masks:
            out["masks"].append({**where, **mask_entry(mask, labels, instances, masks)})
        for li in lf.label_images:
            out["label_images"].append(
                {**where, **label_image_entry(li, labels, instances)}
            )
        for bbox in lf.bboxes:
            out["bboxes"].append({**where, **bbox_entry(bbox, labels, instances)})
        for centroid in lf.centroids:
            out["centroids"].append(
                {**where, **centroid_entry(centroid, labels, instances)}
            )
    for roi in labels.static_rois:
        out["static_rois"].append(
            {
                **roi_entry(roi, labels, instances),
                "video": None if roi.video is None else labels.videos.index(roi.video),
            }
        )
    return out


def attempt(result, key, run):
    """Set `result[key]` to what `run` returns, or `result["error"]` to why it can't."""
    try:
        result[key] = run()
    except Exception as e:  # noqa: BLE001 - the error is the result here.
        result.setdefault("error", {})[key] = f"{type(e).__name__}: {e}"


def resave(path):
    """Save the file again from Python, in the layout Python itself writes."""
    out = f"{path}.resaved.slp"
    sio.save_slp(sio.load_slp(path, open_videos=False), out, embed=False)
    return out


results = {"version": sio.__version__, "files": {}}
for path in sys.argv[1:]:
    result = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        attempt(result, "eager", lambda: read(sio.load_slp(path, open_videos=False)))
        attempt(
            result,
            "lazy",
            lambda: read(sio.load_slp(path, open_videos=False, lazy=True).materialize()),
        )
        attempt(result, "resaved", lambda: resave(path))
    result["warnings"] = sorted({str(w.message) for w in caught})
    results["files"][path] = result
print(json.dumps(results))
