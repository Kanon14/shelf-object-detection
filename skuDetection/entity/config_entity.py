from dataclasses import dataclass

@dataclass
class Facing:
    id: str
    x1: int
    y1: int
    x2: int
    y2: int
    min_count: int = 1
    
@dataclass
class OOSResult:
    facing_id: str
    count: int
    occupancy: float
    status: str # "OK" | "LOW" | "OOS"