from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional
import asyncpg
import aiohttp
import os
from datetime import datetime, timedelta, date
import json
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Database connection pool
async def get_db_pool():
    return await asyncpg.create_pool(
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password"),
        database=os.getenv("DB_NAME", "event_planner"),
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", 5432)
    )

# Pydantic models
class EventCreate(BaseModel):
    name: str
    location: str
    date: date
    event_type: str  # e.g., cricket, wedding, hiking

class Event(EventCreate):
    id: int
    created_at: datetime

class WeatherRequest(BaseModel):
    location: str
    date: str

class WeatherAnalysis(BaseModel):
    temperature: float
    precipitation: float
    wind_speed: float
    cloudiness: str
    suitability_score: str
    details: str

class AlternativeDate(BaseModel):
    date: str
    suitability_score: str

# Simple in-memory cache
weather_cache = {}

# Initialize ML model (trained on mock data for demo)
X_train = np.array([[20, 10, 15, 0], [30, 50, 25, 1], [15, 5, 10, 0]])  # [temp, precip, wind, cloudy]
y_train = np.array([1, 0, 1])  # 1=Good, 0=Poor
ml_model = DecisionTreeClassifier().fit(X_train, y_train)

async def init_db():
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                location VARCHAR(100) NOT NULL,
                date DATE NOT NULL,
                event_type VARCHAR(50) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS event_votes (
                id SERIAL PRIMARY KEY,
                event_id INTEGER REFERENCES events(id),
                user_name VARCHAR(50) NOT NULL,
                preferred_date DATE NOT NULL,
                UNIQUE(event_id, user_name)  -- Prevent duplicate votes by the same user
            );
        """)
    await pool.close()

@app.on_event("startup")
async def startup_event():
    await init_db()

# Weather API integration
async def fetch_weather(location: str, date: str) -> dict:
    cache_key = f"{location}_{date}"
    if cache_key in weather_cache and (datetime.now() - weather_cache[cache_key]["timestamp"]).seconds < 3*3600:
        return weather_cache[cache_key]["data"]

    api_key = os.getenv("OPENWEATHERMAP_API_KEY", "your_api_key_here")
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={location}&appid={api_key}&units=metric"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=400, detail="Invalid location or API error")
            data = await response.json()
            # Match only the date part of dt_txt (e.g., "2025-06-19 12:00:00" -> "2025-06-19")
            forecast = next((item for item in data["list"] if item["dt_txt"].split(" ")[0] == date), None)
            if not forecast:
                raise HTTPException(status_code=404, detail="Weather data not available for date")
            
            weather_data = {
                "temperature": forecast["main"]["temp"],
                "precipitation": forecast["pop"] * 100,
                "wind_speed": forecast["wind"]["speed"] * 3.6,  # Convert m/s to km/h
                "cloudiness": forecast["weather"][0]["description"]
            }
            weather_cache[cache_key] = {"data": weather_data, "timestamp": datetime.now()}
            return weather_data

# Weather scoring logic
def calculate_suitability(event_type: str, weather: dict) -> tuple[str, str]:
    score = 0
    details = []
    
    if event_type.lower() == "cricket":
        if 15 <= weather["temperature"] <= 30:
            score += 30
            details.append("Ideal temperature for cricket")
        if weather["precipitation"] < 20:
            score += 25
            details.append("Low precipitation, good for play")
        if weather["wind_speed"] < 20:
            score += 20
            details.append("Low wind speed, suitable for cricket")
        if "clear" in weather["cloudiness"].lower() or "partly cloudy" in weather["cloudiness"].lower():
            score += 25
            details.append("Clear or partly cloudy skies")
    elif event_type.lower() == "wedding":
        if 18 <= weather["temperature"] <= 28:
            score += 30
            details.append("Comfortable temperature for wedding")
        if weather["precipitation"] < 10:
            score += 30
            details.append("Very low precipitation, ideal for outdoor wedding")
        if weather["wind_speed"] < 15:
            score += 25
            details.append("Low wind, good for decorations")
        if "clear" in weather["cloudiness"].lower():
            score += 15
            details.append("Clear skies for aesthetic appeal")
    
    suitability = "Good" if score >= 70 else "Okay" if score >= 40 else "Poor"
    return suitability, "; ".join(details)

# AI-powered suitability (optional feature)
def calculate_ai_suitability(weather: dict) -> str:
    features = np.array([[weather["temperature"], weather["precipitation"], weather["wind_speed"], 
                         1 if "cloudy" in weather["cloudiness"].lower() else 0]])
    prediction = ml_model.predict(features)[0]
    return "Good" if prediction == 1 else "Poor"

# Event Management Endpoints
@app.post("/events", response_model=Event, status_code=status.HTTP_201_CREATED)
async def create_event(event: EventCreate):
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        event_id = await conn.fetchval(
            """
            INSERT INTO events (name, location, date, event_type)
            VALUES ($1, $2, $3, $4) RETURNING id
            """,
            event.name, event.location, event.date, event.event_type
        )
        return {**event.dict(), "id": event_id, "created_at": datetime.now()}

@app.get("/events", response_model=List[Event])
async def list_events():
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM events")
        return [Event(**row) for row in rows]

@app.put("/events/{event_id}", response_model=Event)
async def update_event(event_id: int, event: EventCreate):
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        result = await conn.fetchval(
            """
            UPDATE events SET name=$1, location=$2, date=$3, event_type=$4
            WHERE id=$5 RETURNING id
            """,
            event.name, event.location, event.date, event.event_type, event_id
        )
        if not result:
            raise HTTPException(status_code=404, detail="Event not found")
        return {**event.dict(), "id": event_id, "created_at": datetime.now()}

# Weather Integration Endpoints
@app.get("/weather/{location}/{date}", response_model=WeatherAnalysis)
async def get_weather(location: str, date: str):
    try:
        weather = await fetch_weather(location, date)
        suitability, details = calculate_suitability("cricket", weather)  # Default to cricket for simplicity
        return WeatherAnalysis(
            temperature=weather["temperature"],
            precipitation=weather["precipitation"],
            wind_speed=weather["wind_speed"],
            cloudiness=weather["cloudiness"],
            suitability_score=suitability,
            details=details
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")

@app.post("/events/{event_id}/weather-check", response_model=WeatherAnalysis)
async def check_event_weather(event_id: int):
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        event = await conn.fetchrow("SELECT * FROM events WHERE id=$1", event_id)
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        event_date_str = event["date"].strftime("%Y-%m-%d")
        weather = await fetch_weather(event["location"], event_date_str)
        suitability, details = calculate_suitability(event["event_type"], weather)
        return WeatherAnalysis(
            temperature=weather["temperature"],
            precipitation=weather["precipitation"],
            wind_speed=weather["wind_speed"],
            cloudiness=weather["cloudiness"],
            suitability_score=suitability,
            details=details
        )

@app.get("/events/{event_id}/suitability", response_model=WeatherAnalysis)
async def get_event_suitability(event_id: int):
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        event = await conn.fetchrow("SELECT * FROM events WHERE id=$1", event_id)
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        event_date_str = event["date"].strftime("%Y-%m-%d")
        weather = await fetch_weather(event["location"], event_date_str)
        suitability, details = calculate_suitability(event["event_type"], weather)
        ai_suitability = calculate_ai_suitability(weather)
        return WeatherAnalysis(
            temperature=weather["temperature"],
            precipitation=weather["precipitation"],
            wind_speed=weather["wind_speed"],
            cloudiness=weather["cloudiness"],
            suitability_score=ai_suitability if ai_suitability == "Good" else suitability,
            details=details
        )

@app.get("/events/{event_id}/alternatives", response_model=List[AlternativeDate])
async def get_alternative_dates(event_id: int):
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        event = await conn.fetchrow("SELECT * FROM events WHERE id=$1", event_id)
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        
        alternatives = []
        base_date = event["date"]
        for i in range(1, 6):  # Check next 5 days
            new_date = base_date + timedelta(days=i)
            new_date_str = new_date.strftime("%Y-%m-%d")
            try:
                weather = await fetch_weather(event["location"], new_date_str)
                suitability, _ = calculate_suitability(event["event_type"], weather)
                alternatives.append(AlternativeDate(date=new_date_str, suitability_score=suitability))
            except HTTPException as e:
                if e.status_code == 404:  # Weather data not available
                    alternatives.append(AlternativeDate(date=new_date_str, suitability_score="Unknown"))
                else:
                    raise e
        return sorted(alternatives, key=lambda x: {"Good": 0, "Okay": 1, "Poor": 2, "Unknown": 3}[x.suitability_score])

# Collaborative Planning (Optional Feature)
@app.post("/events/{event_id}/invite")
async def invite_to_event(event_id: int, user_name: str, preferred_date: str):
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        event = await conn.fetchrow("SELECT * FROM events WHERE id=$1", event_id)
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        preferred_date_obj = datetime.strptime(preferred_date, "%Y-%m-%d").date()
        try:
            await conn.execute(
                """
                INSERT INTO event_votes (event_id, user_name, preferred_date)
                VALUES ($1, $2, $3)
                """,
                event_id, user_name, preferred_date_obj
            )
        except asyncpg.exceptions.UniqueViolationError:
            raise HTTPException(status_code=400, detail=f"{user_name} has already voted for this event")
        return {"success": True, "message": f"Invite sent to {user_name}"}

@app.get("/events/{event_id}/votes")
async def get_event_votes(event_id: int):
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        event = await conn.fetchrow("SELECT * FROM events WHERE id=$1", event_id)
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        votes = await conn.fetch("SELECT user_name, preferred_date FROM event_votes WHERE event_id=$1", event_id)
        return [{"user_name": row["user_name"], "preferred_date": row["preferred_date"].strftime("%Y-%m-%d")} for row in votes]