# Smart Event Planner

A backend service for planning outdoor events with weather-based recommendations using OpenWeatherMap API, enhanced with AI-powered suitability scoring and collaborative event planning.

## Setup Instructions
1. Clone the repository: `git clone <repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables in `.env` (see `.env` example).
4. Run the application: `uvicorn main:app --reload`
5. Deploy to Railway.app for public access.

## Database Schema
- **events**: Stores event details (id, name, location, date, event_type, created_at).
- **event_votes**: Stores collaborative votes (id, event_id, user_name, preferred_date).

## API Documentation
- **POST /events**: Create an event (e.g., `{"name": "Cricket Tournament", "location": "Mumbai", "date": "2025-03-16", "event_type": "cricket"}`).
- **GET /events**: List all events.
- **PUT /events/:id**: Update event details.
- **GET /weather/:location/:date**: Get weather data.
- **POST /events/:id/weather-check**: Analyze weather for an event.
- **GET /events/:id/suitability**: Get AI-enhanced suitability score.
- **GET /events/:id/alternatives**: Suggest alternative dates.
- **POST /events/:id/invite**: Invite users to vote on event dates.
- **GET /events/:id/votes**: View user votes.

## AI Suitability Model
- Uses a decision tree classifier trained on mock weather data (temperature, precipitation, wind, cloudiness).
- Predicts "Good" or "Poor" suitability, enhancing rule-based scoring.

## Known Limitations
- AI model uses mock data; real training data would improve accuracy.
- Weather cache is in-memory; Redis recommended for production.
- Collaborative planning is basic; lacks real-time notifications.

## Postman Collection
- Available at: [GitHub Gist URL]
- Pre-populated with test scenarios (Mumbai cricket, Goa wedding, Lonavala hiking).