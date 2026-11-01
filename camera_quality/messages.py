"""Versioned JSON contract. Measurements are pixels, never physical units."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Region(StrictModel):
    x: Annotated[int, Field(ge=0)]
    y: Annotated[int, Field(ge=0)]
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]
    area_px: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    center_x: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    center_y: Annotated[float, Field(ge=0, allow_inf_nan=False)]


class ClassicalResult(StrictModel):
    decision: Literal["pass", "fail"]
    regions: Annotated[list[Region], Field(max_length=256)]


class CNNResult(StrictModel):
    decision: Literal["pass", "fail"]
    probability: Probability
    threshold: Probability


class InspectionMessage(StrictModel):
    schema_version: Literal[1]
    image_id: Annotated[str, Field(pattern=r"^sim-[a-f0-9]{24}$")]
    camera_id: Literal["synthetic-camera-01"]
    source: Literal["synthetic"]
    captured_at: str
    width: Annotated[int, Field(ge=8, le=4096)]
    height: Annotated[int, Field(ge=8, le=4096)]
    opencv: ClassicalResult
    cnn: CNNResult
    decision: Literal["pass", "fail"]

    @field_validator("schema_version", mode="before")
    @classmethod
    def exact_integer_version(cls, value):
        # Python bool compares equal to 1; it is not a JSON schema version.
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value

    @model_validator(mode="after")
    def coherent(self):
        instant = datetime.fromisoformat(self.captured_at.replace("Z", "+00:00"))
        if instant.tzinfo is None or instant.utcoffset().total_seconds() != 0:
            raise ValueError("captured_at must include UTC timezone")
        if (self.opencv.decision == "fail") != bool(self.opencv.regions):
            raise ValueError("OpenCV decision disagrees with regions")
        if (self.cnn.decision == "fail") != (self.cnn.probability >= self.cnn.threshold):
            raise ValueError("CNN decision disagrees with threshold")
        expected = "fail" if "fail" in (self.opencv.decision, self.cnn.decision) else "pass"
        if self.decision != expected:
            raise ValueError("Combined decision must be fail if either method fails")
        for region in self.opencv.regions:
            if region.x+region.width > self.width or region.y+region.height > self.height:
                raise ValueError("Bounding box outside image")
            if not (region.x <= region.center_x < region.x+region.width and region.y <= region.center_y < region.y+region.height):
                raise ValueError("Centroid outside bounding box")
            if region.area_px > region.width*region.height:
                raise ValueError("Contour area exceeds box area")
        return self


def validate_message(payload: str | bytes) -> InspectionMessage:
    if len(payload) > 64_000:
        raise ValueError("Message exceeds 64 kB")
    return InspectionMessage.model_validate_json(payload)
