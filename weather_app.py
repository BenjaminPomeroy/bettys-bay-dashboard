import json
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import ephem
import pandas as pd
import requests
import streamlit as st


TIMEZONE = ZoneInfo("Africa/Johannesburg")
# Save the forecast alongside this Python file.
WEATHER_CACHE_FILE = (
    Path(__file__).resolve().parent / "saved_weather.json"
)


def validate_weather_data(data):
    """Check the forecast structure before saving or displaying it."""

    if not isinstance(data, dict):
        raise ValueError("Weather data must be a dictionary.")

    for section in ("current", "daily", "hourly"):
        if not isinstance(data.get(section), dict):
            raise ValueError(f"Missing weather section: {section}")

    def check_columns(section_name, fields):
        """Require non-empty lists with matching lengths."""

        section = data[section_name]

        for field in fields:
            values = section.get(field)

            if not isinstance(values, list) or not values:
                raise ValueError(
                    f"Missing or empty field: {section_name}.{field}"
                )

        expected_length = len(section["time"])

        for field in fields:
            if len(section[field]) != expected_length:
                raise ValueError(
                    f"Column length mismatch: {section_name}.{field}"
                )

        for timestamp in section["time"]:
            if not isinstance(timestamp, str):
                raise ValueError(
                    f"Invalid timestamp in {section_name}.time"
                )

            datetime.fromisoformat(timestamp)

    check_columns(
        "daily",
        (
            "time",
            "temperature_2m_min",
            "temperature_2m_max",
            "precipitation_probability_max",
            "wind_speed_10m_max",
            "sunrise",
            "sunset",
        ),
    )

    check_columns(
        "hourly",
        (
            "time",
            "temperature_2m",
            "precipitation_probability",
            "wind_speed_10m",
            "wind_gusts_10m",
            "wind_direction_10m",
            "cloud_cover",
        ),
    )

    # Missing sunrise/sunset readings are allowed, but supplied
    # timestamps must be readable.
    for field in ("sunrise", "sunset"):
        for timestamp in data["daily"][field]:
            if timestamp is not None:
                if not isinstance(timestamp, str):
                    raise ValueError(f"Invalid daily timestamp: {field}")

                datetime.fromisoformat(timestamp)

    current = data["current"]

    for field in (
        "time",
        "temperature_2m",
        "apparent_temperature",
        "wind_speed_10m",
        "cloud_cover",
        "weather_code",
    ):
        if field not in current:
            raise ValueError(f"Missing current field: {field}")

    if not isinstance(current["time"], str):
        raise ValueError("Invalid current conditions timestamp.")

    datetime.fromisoformat(current["time"])


def save_weather_to_disk(data):
    """Save safely, keeping the previous file if writing fails."""

    temporary_file = WEATHER_CACHE_FILE.with_suffix(".tmp")

    try:
        text = json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

        temporary_file.write_text(text, encoding="utf-8")

        # Replace the old file only after the new file has been written.
        temporary_file.replace(WEATHER_CACHE_FILE)

        return True

    except (OSError, TypeError, ValueError):
        return False


def load_weather_from_disk():
    """Read a saved forecast, returning None if it is missing or invalid."""

    try:
        text = WEATHER_CACHE_FILE.read_text(encoding="utf-8")
        data = json.loads(text)

        validate_weather_data(data)

        downloaded_at = datetime.fromisoformat(data["_downloaded_at"])

        if downloaded_at.tzinfo is None:
            raise ValueError("Saved download time has no timezone.")

        return data

    except (OSError, ValueError, KeyError, TypeError):
        return None

st.set_page_config(
    page_title="Betty's Bay Dashboard",
    page_icon="🌊",
    layout="wide",
)

@st.cache_data(ttl=1800, show_spinner=False)
def get_weather(forecast_day):
    """Cache weather for 30 minutes, separately for each local date."""

    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": -34.36,
            "longitude": 18.90,
            "current": (
                "temperature_2m,"
                "apparent_temperature,"
                "wind_speed_10m,"
                "wind_direction_10m,"
                "wind_gusts_10m,"
                "cloud_cover,"
                "weather_code"
            ),
             "hourly": (
                "temperature_2m,"
                "precipitation_probability,"
                "wind_speed_10m,"
                "wind_gusts_10m,"
                "wind_direction_10m,"
                "cloud_cover"
            ),
             "daily": (
                "temperature_2m_max,"
                "temperature_2m_min,"
                "precipitation_probability_max,"
                "wind_speed_10m_max,"
                "sunrise,"
                "sunset"
            ),
            "timezone": "Africa/Johannesburg",
            "forecast_days": 10,
        },
        timeout=20,
    )
    response.raise_for_status()

    data = response.json()
    validate_weather_data(data)

    forecast_start = datetime.fromisoformat(
        data["daily"]["time"][0]
    ).date().isoformat()

    if forecast_start != forecast_day:
        raise ValueError(
            "The downloaded forecast does not start on the requested date."
        )

    data["_downloaded_at"] = datetime.now(TIMEZONE).isoformat()

    saved_successfully = save_weather_to_disk(data)

    # This flag is for the interface, not part of the saved forecast.
    data["_save_failed"] = not saved_successfully

    return data

@st.cache_data(ttl=3600, show_spinner=False)
def get_sea_temperature():
    """Get a modelled sea temperature at a nearby offshore point."""

    response = requests.get(
        "https://marine-api.open-meteo.com/v1/marine",
        params={
            # Offshore, rather than a point on land.
            "latitude": -34.40,
            "longitude": 18.90,
            "hourly": "sea_surface_temperature",
            "timezone": "Africa/Johannesburg",
            "forecast_days": 1,
            "cell_selection": "sea",
        },
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()

    hourly = data["hourly"]
    readings = []

    for timestamp, temperature in zip(
        hourly["time"],
        hourly["sea_surface_temperature"],
    ):
        if temperature is not None:
            reading_time = datetime.fromisoformat(timestamp)
            reading_time = reading_time.replace(tzinfo=TIMEZONE)
            readings.append((reading_time, temperature))

    if not readings:
        raise ValueError("No sea-temperature data available.")

    now = datetime.now(TIMEZONE)

    # Choose the available reading closest to the current time.
    return min(
        readings,
        key=lambda reading: abs((reading[0] - now).total_seconds()),
    )


def format_value(value, unit=""):
    """Display missing values clearly instead of calling them zero."""
    if value is None:
        return "Unavailable"
    return f"{value}{unit}"


def show_forecast(weather, number_of_days):
    """Turn the API data into a readable forecast table."""

    daily = weather["daily"]
    hourly = weather["hourly"]

    # Group hourly cloud coverage by date.
    clouds_by_date = {}

    for timestamp, coverage in zip(
        hourly["time"],
        hourly["cloud_cover"],
    ):
        if coverage is not None:
            date = timestamp[:10]
            clouds_by_date.setdefault(date, []).append(coverage)

    rows = []

    for index, date in enumerate(daily["time"][:number_of_days]):
        cloud_values = clouds_by_date.get(date, [])
        average_cloud = (
            round(sum(cloud_values) / len(cloud_values))
            if cloud_values else None
        )

        rows.append({
            "Day": datetime.fromisoformat(date).strftime("%a %d %b"),
            "Min °C": daily["temperature_2m_min"][index],
            "Max °C": daily["temperature_2m_max"][index],
            "Rain probability %": (
                daily["precipitation_probability_max"][index]
            ),
            "Average cloud %": average_cloud,
            "Max wind km/h": daily["wind_speed_10m_max"][index],
        })

    st.dataframe(
        rows,
        hide_index=True,
        use_container_width=True,
    )
    with st.expander("Show forecast charts"):
        # Convert our list of daily rows into a pandas table.
        chart_data = pd.DataFrame(rows)

        # Use actual dates so the charts have a chronological time axis.
        chart_data["Date"] = pd.to_datetime(
            daily["time"][:number_of_days]
        )

        chart_data = chart_data.set_index("Date")

        st.markdown("### 🌡️ Daily temperatures")

        st.line_chart(
            chart_data[["Min °C", "Max °C"]],
            height=260,
        )

        st.caption(
            "Forecast minimum and maximum for each day, "
            "not an hour-by-hour temperature curve."
        )

        st.markdown("### 🌧️ Daily rain probability")

        st.bar_chart(
            chart_data[["Rain probability %"]],
            height=230,
        )

        st.caption(
            "The maximum hourly precipitation probability for each day. "
            "This is not the expected amount of rain."
        )

    st.caption(
        "Includes today. Cloud cover is the average of the hourly "
        "forecast values. Longer-range forecasts are less certain."
    )


def local_moon_date(date):
    """Convert an Ephem UTC date to South African time."""

    utc_date = date.datetime().replace(tzinfo=timezone.utc)

    return utc_date.astimezone(TIMEZONE).strftime(
        "%a %d %b %Y at %H:%M"
    )


def show_moon():
    now = ephem.now()
    moon = ephem.Moon(now)

    next_new = ephem.next_new_moon(now)
    next_full = ephem.next_full_moon(now)

    # Between new moon and full moon, illumination is increasing.
    waxing = next_full < next_new

    st.subheader("The Moon today")

    illumination_column, cycle_column = st.columns(2)

    illumination_column.metric(
        "Illuminated",
        f"{moon.phase:.1f}%",
    )

    cycle_column.metric(
        "Cycle",
        "Waxing" if waxing else "Waning",
    )

    st.write(
        "**Waxing** means the illuminated portion is increasing. "
        "**Waning** means it is decreasing."
    )

    events = [
        ("New moon", next_new),
        ("First quarter", ephem.next_first_quarter_moon(now)),
        ("Full moon", next_full),
        ("Last quarter", ephem.next_last_quarter_moon(now)),
    ]

    events.sort(key=lambda event: float(event[1]))

    st.subheader("Upcoming lunar phases")

    st.table([
        {
            "Phase": name,
            "South African time": local_moon_date(date),
        }
        for name, date in events
    ])

    st.caption(
        "Calculated locally using PyEphem. "
        "Lunar phases are not predictions of local tide times."
    )


# -----------------------
# Page layout
# -----------------------

def describe_weather(code):
    """Convert an Open-Meteo weather code into an icon and description."""

    conditions = {
        0: ("☀️", "Clear sky"),
        1: ("🌤️", "Mainly clear"),
        2: ("⛅", "Partly cloudy"),
        3: ("☁️", "Overcast"),
        45: ("🌫️", "Fog"),
        48: ("🌫️", "Freezing fog"),
        51: ("🌦️", "Light drizzle"),
        53: ("🌦️", "Moderate drizzle"),
        55: ("🌧️", "Heavy drizzle"),
        56: ("🌧️", "Light freezing drizzle"),
        57: ("🌧️", "Heavy freezing drizzle"),
        61: ("🌧️", "Light rain"),
        63: ("🌧️", "Moderate rain"),
        65: ("🌧️", "Heavy rain"),
        66: ("🌧️", "Light freezing rain"),
        67: ("🌧️", "Heavy freezing rain"),
        71: ("🌨️", "Light snow"),
        73: ("🌨️", "Moderate snow"),
        75: ("🌨️", "Heavy snow"),
        77: ("🌨️", "Snow grains"),
        80: ("🌦️", "Light rain showers"),
        81: ("🌧️", "Moderate rain showers"),
        82: ("🌧️", "Violent rain showers"),
        85: ("🌨️", "Light snow showers"),
        86: ("🌨️", "Heavy snow showers"),
        95: ("⛈️", "Thunderstorm"),
        96: ("⛈️", "Thunderstorm with slight hail"),
        99: ("⛈️", "Thunderstorm with heavy hail"),
    }

    return conditions.get(code, ("🌡️", "Conditions unavailable"))

def compass_direction(degrees):
    """Convert a wind bearing into a compass direction."""

    if pd.isna(degrees):
        return "Unavailable"

    directions = [
        "N", "NNE", "NE", "ENE",
        "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW",
        "W", "WNW", "NW", "NNW",
    ]

    index = int((float(degrees) + 11.25) / 22.5) % 16

    return directions[index]

def show_today_summary(weather):
    """Display a compact overview without hiding missing information."""

    current = weather.get("current", {})
    daily = weather.get("daily", {})

    icon, description = describe_weather(current.get("weather_code"))

    def first_value(name):
        values = daily.get(name)
        return values[0] if values else None

    with st.container(border=True):
        st.subheader(f"{icon} {description}")
        st.caption(
            "Modelled conditions for the time shown below. "
            "If a saved-forecast warning is displayed, these are not fresh data."
        )

        low, high = st.columns(2)
        rain, wind = st.columns(2)

        low.metric(
            "Today's minimum",
            format_value(first_value("temperature_2m_min"), " °C"),
        )

        high.metric(
            "Today's maximum",
            format_value(first_value("temperature_2m_max"), " °C"),
        )

        rain.metric(
            "Today's rain probability",
            format_value(
                first_value("precipitation_probability_max"),
                "%",
            ),
        )

        wind.metric(
            "Today's maximum wind",
            format_value(first_value("wind_speed_10m_max"), " km/h"),
        )

def show_daylight(weather):
    """Display today's sunrise, sunset and daylight duration."""

    st.subheader("☀️ Daylight today")

    try:
        daily = weather["daily"]

        sunrise = datetime.fromisoformat(daily["sunrise"][0])
        sunset = datetime.fromisoformat(daily["sunset"][0])

        daylight_minutes = int(
            (sunset - sunrise).total_seconds() / 60
        )

        hours, minutes = divmod(daylight_minutes, 60)

        sunrise_column, sunset_column = st.columns(2)

        sunrise_column.metric("Sunrise", sunrise.strftime("%H:%M"))
        sunset_column.metric("Sunset", sunset.strftime("%H:%M"))

        st.caption(
            f"Daylight: {hours}h {minutes}m · South African time."
        )

    except (KeyError, IndexError, TypeError, ValueError):
        st.info("Today's daylight information is unavailable.")

def show_weather_highlights(weather):
    """Summarise the next 24 hourly forecast slots."""

    st.subheader("⏱️ Next 24 hours at a glance")

    try:
        hourly = weather["hourly"]

        forecast = pd.DataFrame({
            "Time": pd.to_datetime(hourly["time"]),
            "Temperature": hourly["temperature_2m"],
            "Rain": hourly["precipitation_probability"],
            "Gusts": hourly["wind_gusts_10m"],
        })

        # Match the API's local timestamps, which have no timezone suffix.
        current_hour = datetime.now(TIMEZONE).replace(
            minute=0,
            second=0,
            microsecond=0,
            tzinfo=None,
        )

        end_hour = current_hour + pd.Timedelta(hours=24)

        upcoming = forecast.loc[
            (forecast["Time"] >= current_hour)
            & (forecast["Time"] < end_hour)
        ].copy()

        if upcoming.empty:
            st.info("No upcoming hourly forecast is available.")
            return

        if len(upcoming) < 24:
            st.caption(
                f"Only {len(upcoming)} hourly forecast slots are available "
                "in this 24-hour window."
            )

        highlights = [
            ("Gusts", "💨 Peak forecast gust", " km/h"),
            ("Rain", "🌧️ Highest hourly rain probability", "%"),
            ("Temperature", "🌡️ Warmest forecast hour", " °C"),
        ]

        # Short horizontal cards keep values and times together.
        for field, label, unit in highlights:
            values = pd.to_numeric(
                upcoming[field],
                errors="coerce",
            ).dropna()

            with st.container(border=True):
                if values.empty:
                    st.metric(label, "Unavailable")

                else:
                    peak_index = values.idxmax()
                    peak_value = values.loc[peak_index]
                    peak_time = upcoming.loc[peak_index, "Time"]

                    st.metric(label, f"{peak_value:g}{unit}")

                    st.caption(
                        peak_time.strftime("%a %d %b at %H:%M")
                        + " SAST"
                    )

        st.caption(
            "Starting at the current hour. Highlights use available "
            "readings; ties show the earliest forecast hour. "
            "Rain probability is not rainfall amount."
        )

    except (KeyError, TypeError, ValueError):
        st.info("Weather highlights are currently unavailable.")


st.title("🌊 Betty's Bay Dashboard")

local_now = datetime.now(TIMEZONE)

st.write(
    local_now.strftime("%A, %d %B %Y")
    + " · Betty's Bay, Western Cape"
)

st.caption(
    "Your local weather and coastal information dashboard. "
    "All displayed times are South African time."
)

if st.button("Refresh weather data"):
    get_weather.clear()
    get_sea_temperature.clear()

(
    today_tab,
    hourly_tab,
    three_day_tab,
    ten_day_tab,
    moon_tab,
    tides_tab,
) = st.tabs([
    "🌤️ Today",
    "⏱️ Next 24 hours",
    "📅 3-day forecast",
    "🗓️ 10-day forecast",
    "🌙 Lunar cycles",
    "🌊 Tides",
])

weather = None
using_saved_weather = False
older_saved_weather = None

try:
    with st.spinner("Fetching weather..."):
        weather = get_weather(
            datetime.now(TIMEZONE).date().isoformat()
        )

except (requests.RequestException, ValueError, KeyError, TypeError):
    saved_weather = load_weather_from_disk()

    if saved_weather is None:
        st.error(
            "Couldn't download the weather, and there is no usable "
            "saved forecast yet. Connect to the internet and click "
            "Refresh weather data."
        )

    else:
        downloaded_time = datetime.fromisoformat(
            saved_weather["_downloaded_at"]
        ).astimezone(TIMEZONE)

        downloaded_label = downloaded_time.strftime(
            "%d %b %Y at %H:%M"
        )

        forecast_start_date = datetime.fromisoformat(
            saved_weather["daily"]["time"][0]
        ).date()

        today = datetime.now(TIMEZONE).date()

        if forecast_start_date == today:
            weather = saved_weather
            using_saved_weather = True

            st.warning(
                "SAVED FORECAST — a fresh download failed. "
                f"Showing weather downloaded on {downloaded_label} SAST. "
                "Conditions may have changed since that download."
            )

        else:
            # Do not fill the Today tab with another day's conditions.
            older_saved_weather = saved_weather

            st.warning(
                "A fresh download failed. Your saved forecast was "
                f"downloaded on {downloaded_label} SAST and starts on "
                f"{forecast_start_date.strftime('%d %b %Y')}. "
                "It is shown separately below, not as today's conditions."
            )


if weather is not None and weather.get("_save_failed"):
    st.warning(
        "Weather downloaded successfully, but could not be saved to disk. "
        "Check that the app folder is writable and your disk has free space."
    )


if older_saved_weather is not None:
    with st.expander("View the older saved forecast", expanded=True):
        daily = older_saved_weather["daily"]

        saved_rows = []

        for date, minimum, maximum, rain in zip(
            daily["time"],
            daily["temperature_2m_min"],
            daily["temperature_2m_max"],
            daily["precipitation_probability_max"],
        ):
            saved_rows.append({
                "Forecast date": date,
                "Minimum °C": minimum,
                "Maximum °C": maximum,
                "Rain probability %": rain,
            })

        st.dataframe(
            saved_rows,
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "This is an older forecast snapshot, not a new prediction "
            "or a record of what actually happened. "
            "It has not been updated using current conditions."
        )



# Show when the weather was actually downloaded.
# This is different from the time represented by a forecast reading.
if weather is not None:
    downloaded_at = weather.get("_downloaded_at")

    if downloaded_at:
        try:
            downloaded_time = datetime.fromisoformat(downloaded_at)

            age_minutes = max(
                0,
                int(
                    (
                        datetime.now(TIMEZONE) - downloaded_time
                    ).total_seconds() / 60
                ),
            )

            age_text = (
                "less than a minute ago"
                if age_minutes == 0
                else f"{age_minutes} minute(s) ago"
            )

            st.caption(
                "Weather downloaded: "
                + downloaded_time.strftime("%d %b at %H:%M")
                + f" SAST · {age_text}"
            )

        except (TypeError, ValueError):
            st.caption("Weather download time is unavailable.")

    st.caption(
        "Weather requests are cached for 30 minutes. "
        "Use Refresh weather data to request a fresh download."
    )


with today_tab:
    if weather is not None:
        show_today_summary(weather)

    st.subheader(
        "Saved conditions"
        if using_saved_weather
        else "Current conditions"
    )

    if weather is not None:
        try:
            current = weather["current"]

            columns = st.columns(3) + st.columns(3)

            columns[0].metric(
                "Air temperature",
                format_value(current["temperature_2m"], " °C"),
            )

            columns[1].metric(
                "Feels like",
                format_value(current["apparent_temperature"], " °C"),
            )

            wind_from = compass_direction(
                current.get("wind_direction_10m")
            )

            wind_label = (
                "Wind"
                if wind_from == "Unavailable"
                else f"Wind from {wind_from}"
            )

            columns[3].metric(
                wind_label,
                format_value(current.get("wind_speed_10m"), " km/h"),
            )
            columns[2].metric(
                "Cloud coverage",
                format_value(current["cloud_cover"], "%"),
            )

            columns[4].metric(
                "Wind gusts",
                format_value(current.get("wind_gusts_10m"), " km/h"),
            )

            st.caption(
                "Wind direction tells you where the wind comes FROM. "
                "Gusts are brief increases above the sustained wind."
            )

            st.caption(
                "Conditions time: "
                + current["time"].replace("T", " at ")
                + " — South African time."
            )

        except (KeyError, TypeError):
            st.warning("Some current weather information is unavailable.")

    if weather is not None:
        st.divider()
        show_daylight(weather)

        st.divider()
        show_weather_highlights(weather)

    st.divider()

    st.subheader("Water temperature")

    try:
        with st.spinner("Fetching sea temperature..."):
            sea_time, sea_temperature = get_sea_temperature()

        st.metric(
            "Offshore sea-surface temperature",
            f"{sea_temperature:.1f} °C",
        )

        st.caption(
            "Modelled offshore estimate near Betty's Bay, not a "
            "measurement at your swimming beach. "
            f"Estimate time: {sea_time.strftime('%d %b at %H:%M')} SAST."
        )

    except (requests.RequestException, ValueError, KeyError, TypeError):
        st.info(
            "Sea-temperature data is currently unavailable from the provider."
        )

with hourly_tab:
    st.subheader("Your next 24 hours")
    st.caption(
        "Forecast starting at the current hour. "
        "All times are South African time."
    )

    if weather is None:
        st.info("Weather is unavailable. Try Refresh weather data.")

    else:
        try:

            # ------------------------------------
            # Organise hourly data into a table
            # ------------------------------------

            hourly = weather["hourly"]

            forecast = pd.DataFrame({
                "Time": pd.to_datetime(hourly["time"]),
                "Temperature °C": hourly["temperature_2m"],
                "Rain probability %": hourly["precipitation_probability"],
                "Wind km/h": hourly["wind_speed_10m"],
                "Gusts km/h": hourly["wind_gusts_10m"],
                "Wind direction °": hourly["wind_direction_10m"],
                "Cloud cover %": hourly["cloud_cover"],
            })

            # The API supplies local times without a timezone suffix.
            # Compare them with the current local hour in the same format.
            current_hour = datetime.now(TIMEZONE).replace(
                minute=0,
                second=0,
                microsecond=0,
                tzinfo=None,
            )

            upcoming = forecast.loc[
                forecast["Time"] >= current_hour
            ].head(24).copy()

            if upcoming.empty:
                st.warning(
                    "No upcoming hourly data was found. "
                    "Try refreshing the weather."
                )

            else:
                chart_data = upcoming.set_index("Time")

                # ------------------------------------
                # Display charts separately because
                # each uses a different measurement.
                # ------------------------------------

                st.markdown("### 🌡️ Temperature")
                st.caption("Degrees Celsius")

                st.line_chart(
                    chart_data[["Temperature °C"]],
                    height=230,
                )

                st.markdown("### 🌧️ Rain probability")
                st.caption(
                    "Chance of precipitation for each hour—not rainfall amount."
                )

                st.bar_chart(
                    chart_data[["Rain probability %"]],
                    height=230,
                )

                st.markdown("### 💨 Wind and gusts")
                st.caption(
                    "Kilometres per hour. "
                    "Gusts are brief increases above the sustained wind."
                )

                st.line_chart(
                    chart_data[["Wind km/h", "Gusts km/h"]],
                    height=230,
                )

                # ------------------------------------
                # Convert wind bearings into compass
                # directions for the detailed table.
                # ------------------------------------


                details = upcoming.copy()

                details["Wind from"] = details[
                    "Wind direction °"
                ].apply(compass_direction)

                details["Time"] = details["Time"].dt.strftime(
                    "%a %d %b %H:%M"
                )

                details = details[[
                    "Time",
                    "Temperature °C",
                    "Rain probability %",
                    "Wind km/h",
                    "Gusts km/h",
                    "Wind from",
                    "Cloud cover %",
                ]]

                with st.expander("Show the hourly details"):
                    st.dataframe(
                        details,
                        hide_index=True,
                        use_container_width=True,
                    )

                    st.caption(
                        "Wind direction describes where the wind comes FROM. "
                        "For example, SE means a southeasterly wind."
                    )

                st.caption(
                    "These are model forecasts, not guarantees or official "
                    "weather warnings. Missing readings are not treated as zero."
                )

        except (KeyError, IndexError, TypeError, ValueError):
            st.warning(
                "Some hourly forecast information is missing or unexpected. "
                "Try Refresh weather data."
            )


with three_day_tab:
    st.subheader("Three-day forecast")

    if weather is not None:
        try:
            show_forecast(weather, 3)
        except (KeyError, IndexError, TypeError):
            st.warning("Forecast data is incomplete.")


with ten_day_tab:
    st.subheader("Ten-day forecast")

    if weather is not None:
        try:
            show_forecast(weather, 10)
        except (KeyError, IndexError, TypeError):
            st.warning("Forecast data is incomplete.")


with moon_tab:
    show_moon()


with tides_tab:
    st.subheader("Tide information")

    st.info(
        "This version links to an official tide-information source. "
        "It does not yet calculate or display Betty's Bay tide times."
    )

    st.link_button(
        "Open SA Navy Hydrographic Office",
        "https://www.sanho.co.za/",
    )

    st.write(
        "On that website, look for tide tables and check the applicable "
        "location, date, time zone and any local corrections."
    )

    st.warning(
        "Do not use a nearby station's predictions as exact local tide times. "
        "Weather and coastal conditions can also affect water levels. "
        "This dashboard is not a navigation or swimming-safety service."
    )


st.divider()
st.caption(
    "Weather and marine data: Open-Meteo. "
    "Weather is cached for 30 minutes; sea temperature for one hour. "
    "Current conditions are model-based estimates."
)
st.markdown("[Weather data source: Open-Meteo](https://open-meteo.com/)")