"""
Vehicle configuration API router.

View and edit vehicle profiles, G-force limits, transmission ratios.
"""

from fastapi import APIRouter, HTTPException, Request

from src.config.vehicles import get_vehicle_database, get_vehicle, set_active_vehicle

router = APIRouter()


@router.get("/api/vehicles")
async def get_vehicles():
    """Get all vehicles and active vehicle ID"""
    db = get_vehicle_database()
    vehicles = db.list_vehicles()
    active_id = db.get_active_vehicle_id()

    return {
        "active_vehicle": active_id,
        "vehicles": [v.to_dict() for v in vehicles]
    }


@router.put("/api/vehicles/active")
async def set_active_vehicle_endpoint(request: Request):
    """Set the active vehicle"""
    data = await request.json()
    vehicle_id = data.get("vehicle_id")

    if not vehicle_id:
        raise HTTPException(status_code=400, detail="vehicle_id is required")

    vehicle = get_vehicle(vehicle_id)
    if vehicle is None:
        raise HTTPException(status_code=404, detail=f"Vehicle not found: {vehicle_id}")

    try:
        set_active_vehicle(vehicle_id)
        return {"status": "ok", "active_vehicle": vehicle_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set active vehicle: {str(e)}")


@router.get("/api/vehicles/{vehicle_id}")
async def get_vehicle_by_id(vehicle_id: str):
    """Get a specific vehicle by ID"""
    vehicle = get_vehicle(vehicle_id)
    if vehicle is None:
        raise HTTPException(status_code=404, detail=f"Vehicle not found: {vehicle_id}")
    return vehicle.to_dict()


@router.put("/api/vehicles/{vehicle_id}")
async def update_vehicle(vehicle_id: str, request: Request):
    """Update a vehicle's parameters"""
    db = get_vehicle_database()

    vehicle = get_vehicle(vehicle_id)
    if vehicle is None:
        raise HTTPException(status_code=404, detail=f"Vehicle not found: {vehicle_id}")

    data = await request.json()

    try:
        db.update_vehicle(vehicle_id, data)
        return {"status": "ok", "message": f"Vehicle {vehicle_id} updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update vehicle: {str(e)}")
