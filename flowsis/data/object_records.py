from typing import Any


def get_object_records(example: dict[str, Any]) -> list[dict[str, Any]]:
    objects = example.get("objects")
    if objects is None:
        return []
    if not isinstance(objects, list):
        raise TypeError(
            f"Expected example['objects'] to be a list, received {type(objects).__name__}."
        )
    return [dict(record) for record in objects]


def get_object_feature_schema(objects_feature: Any) -> Any:
    schema = getattr(objects_feature, "feature", objects_feature)
    if isinstance(schema, list):
        if len(schema) != 1:
            raise TypeError(
                f"Expected a single object feature schema, received {len(schema)} entries."
            )
        return schema[0]
    return schema


def resolve_object_source(
    example: dict[str, Any],
    *,
    object_index: int | None = None,
) -> tuple[int, int]:
    records = get_object_records(example)
    if object_index is not None:
        if object_index < 0 or object_index >= len(records):
            raise IndexError(
                f"Object index {object_index} is out of bounds for {len(records)} objects."
            )
        record = records[object_index]
        return int(record["video_id"]), int(record["frame_idx"])

    if len(records) == 1:
        record = records[0]
        return int(record["video_id"]), int(record["frame_idx"])

    source_pairs = {(record["video_id"], record["frame_idx"]) for record in records}
    if len(source_pairs) == 1:
        video_id, frame_idx = next(iter(source_pairs))
        return int(video_id), int(frame_idx)

    raise KeyError("Could not resolve a unique object source from the example.")
