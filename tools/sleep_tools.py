import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests
from phi.tools import Toolkit
from phi.utils.log import logger


@dataclass
class VaultConfig:
    """Configuration for Vault API connection"""
    base_url: str = "https://vault-api.anlyst.ai"
    token: Optional[str] = None
    
    def __post_init__(self) -> None:
        self.token = self.token or os.getenv("VAULT_API_KEY")
        if not self.token:
            logger.error("No Vault API token provided")

class SleepAPITools(Toolkit):
    """Toolkit for analyzing sleep data from Vault API"""
    
    def __init__(self, api_key: Optional[str] = None) -> None:
        super().__init__(name="sleep_api_tools")
        self.config = VaultConfig(token=api_key)
        self.register(self.get_sleep_data)
        self.register(self.get_sleep_analysis)
        self.register(self.get_recent_sleep_trends)

    def get_sleep_data(self, days: int = 7, next_token: Optional[str] = None) -> str:
        """
        Fetch sleep data for the specified number of days
        
        Args:
            days: Number of days of sleep data to fetch
            next_token: Pagination token for fetching next page
        """
        if not self.config.token:
            return self.format_error("No API token configured")
            
        try:
            params = {}
            if next_token:
                params["next_token"] = next_token
                
            logger.debug(f"Fetching sleep data with params: {params}")
            response = requests.get(
                f"{self.config.base_url}/entries/by_key/sleep",
                headers={"Authorization": f"Bearer {self.config.token}"},
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            # Process and format the data
            if data.get("status") == "success" and data.get("data"):
                sleep_entries = data["data"]
                recent_data = sleep_entries[:days]
                
                # Format the entries for better readability
                formatted_entries = []
                for entry in recent_data:
                    try:
                        formatted_entry = {
                            "date": entry["start_time"].split("T")[0],
                            "start_time": entry["start_time"],
                            "end_time": entry["end_time"],
                            "duration_minutes": round(float(entry.get("duration", 0)) / 60, 2),
                            "quality": int(entry.get("quality", 0)) if entry.get("quality") is not None else None,
                            "respiratory_rate": round(float(entry.get("respiratory_rate", 0)), 2) if entry.get("respiratory_rate") is not None else None,
                            "metadata": {
                                "fragment_id": entry.get("fragment_id"),
                                "source_id": entry.get("source_id"),
                                "provider_id": entry.get("provider_id"),
                                "created_at": entry.get("created_at")
                            }
                        }
                        formatted_entries.append(formatted_entry)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Skipping malformed entry: {e}")
                        continue
                
                return json.dumps({
                    "recent_sleep_data": formatted_entries,
                    "days_analyzed": len(formatted_entries),
                    "total_entries": data["pagination"]["total"],
                    "next_token": data["pagination"].get("next"),
                    "metadata": {
                        "fragment_type": "sleep",
                        "timestamp": datetime.now().isoformat()
                    }
                })
            elif data.get("status") != "success":
                return self.format_error(f"API Error: {data.get('message', 'Unknown error')}")
            else:
                return self.format_error("No sleep data available")
            
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return self.format_error(f"Unexpected error: {str(e)}")

    def get_all_sleep_data(self, max_pages: int = 5) -> str:
        """
        Fetch all available sleep data using pagination
        
        Args:
            max_pages: Maximum number of pages to fetch
        """
        all_entries = []
        next_token = None
        pages_fetched = 0
        
        try:
            while pages_fetched < max_pages:
                raw_data = self.get_sleep_data(days=100, next_token=next_token)
                data = json.loads(raw_data)
                
                if "error" in data:
                    return self.format_error(data["error"])
                
                entries = data.get("recent_sleep_data", [])
                all_entries.extend(entries)
                
                next_token = data.get("next_token")
                if not next_token:
                    break
                    
                pages_fetched += 1
            
            return json.dumps({
                "sleep_data": all_entries,
                "total_entries": len(all_entries),
                "pages_fetched": pages_fetched + 1,
                "has_more": bool(next_token),
                "metadata": {
                    "fragment_type": "sleep",
                    "timestamp": datetime.now().isoformat()
                }
            })
            
        except Exception as e:
            return self.format_error(f"Failed to fetch all sleep data: {str(e)}")

    def get_sleep_analysis(self) -> str:
        """Analyze the most recent sleep entry"""
        try:
            raw_data = self.get_sleep_data(days=1)
            data = json.loads(raw_data)
            
            if "error" in data:
                return self.format_error(data["error"])
                
            recent_sleep = data.get("recent_sleep_data", [])
            if not recent_sleep:
                return self.format_error("No recent sleep data available")
                
            latest_entry = recent_sleep[0]
            
            # Add analysis
            analysis = {
                "latest_sleep": latest_entry,
                "analysis": {
                    "sleep_duration_hours": round(latest_entry["duration_minutes"] / 60, 2),
                    "quality_score": latest_entry.get("quality"),
                    "quality_category": self._get_quality_category(latest_entry.get("quality")),
                    "respiratory_rate": latest_entry.get("respiratory_rate")
                },
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            return json.dumps(analysis)
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return self.format_error(f"Analysis failed: {str(e)}")

    def _get_quality_category(self, quality_score: Optional[int]) -> str:
        """Convert quality score to category"""
        if quality_score is None:
            return "Unknown"
        if quality_score >= 80:
            return "Excellent"
        elif quality_score >= 60:
            return "Good"
        elif quality_score >= 40:
            return "Fair"
        else:
            return "Poor"

    def get_recent_sleep_trends(self, days: int = 7, use_pagination: bool = True) -> str:
        """
        Analyze sleep trends over the specified period
        
        Args:
            days: Number of days to analyze
            use_pagination: Whether to fetch all available data using pagination
        """
        try:
            if use_pagination:
                raw_data = self.get_all_sleep_data(max_pages=10)  # Fetch more historical data
            else:
                raw_data = self.get_sleep_data(days=days)
                
            data = json.loads(raw_data)
            
            if "error" in data:
                return self.format_error(data["error"])
            
            # Get sleep entries from either pagination or direct response
            sleep_entries = data.get("sleep_data", []) if use_pagination else data.get("recent_sleep_data", [])
            if not sleep_entries:
                return self.format_error("No sleep data available for trend analysis")
            
            # Calculate comprehensive trends
            quality_scores = [entry["quality"] for entry in sleep_entries if entry.get("quality") is not None]
            durations = [entry["duration_minutes"] for entry in sleep_entries if entry.get("duration_minutes")]
            respiratory_rates = [entry["respiratory_rate"] for entry in sleep_entries if entry.get("respiratory_rate") is not None]
            
            trends = {
                "period_analyzed": f"Last {len(sleep_entries)} entries",
                "entries_analyzed": len(sleep_entries),
                "summary_stats": {
                    "sleep_duration": {
                        "average_hours": round(sum(durations) / len(durations) / 60, 2) if durations else None,
                        "min_hours": round(min(durations) / 60, 2) if durations else None,
                        "max_hours": round(max(durations) / 60, 2) if durations else None
                    },
                    "sleep_quality": {
                        "average_score": round(sum(quality_scores) / len(quality_scores), 2) if quality_scores else None,
                        "best_quality": max(quality_scores) if quality_scores else None,
                        "lowest_quality": min(quality_scores) if quality_scores else None
                    },
                    "respiratory_rate": {
                        "average": round(sum(respiratory_rates) / len(respiratory_rates), 2) if respiratory_rates else None,
                        "min": round(min(respiratory_rates), 2) if respiratory_rates else None,
                        "max": round(max(respiratory_rates), 2) if respiratory_rates else None
                    }
                },
                "daily_data": sleep_entries,
                "metadata": {
                    "analysis_timestamp": datetime.now().isoformat(),
                    "data_points": len(sleep_entries),
                    "using_pagination": use_pagination
                }
            }
            
            # Add trend indicators
            if len(durations) > 1:
                duration_trend = "improving" if durations[-1] > durations[0] else "declining"
                trends["trends"] = {
                    "duration_trend": duration_trend,
                    "quality_trend": "improving" if quality_scores[-1] > quality_scores[0] else "declining" if quality_scores else "unknown",
                    "consistency_score": self._calculate_consistency_score(durations)
                }
            
            return json.dumps(trends)
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {str(e)}")
            return self.format_error(f"Trend analysis failed: {str(e)}")

    def _calculate_consistency_score(self, durations: List[float]) -> float:
        """Calculate a sleep consistency score based on duration variations"""
        if not durations or len(durations) < 2:
            return 0.0
            
        # Calculate standard deviation of sleep durations
        mean_duration = sum(durations) / len(durations)
        variance = sum((x - mean_duration) ** 2 for x in durations) / len(durations)
        std_dev = variance ** 0.5
        
        # Convert to a 0-100 score (lower variation = higher score)
        max_acceptable_std_dev = 120  # 2 hours in minutes
        consistency_score = 100 * (1 - min(std_dev / max_acceptable_std_dev, 1))
        
        return round(consistency_score, 2)

    def format_error(self, message: str) -> str:
        """Format error messages consistently"""
        return json.dumps({"error": message})