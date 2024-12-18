import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from openai import OpenAI
from phi.tools import Toolkit
from phi.utils.log import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, PointStruct, VectorParams


@dataclass
class VaultConfig:
    """Configuration for Vault API connection"""
    base_url: str = "https://vault-api.anlyst.ai"
    token: Optional[str] = None
    collection_name: str = "sleep_data"
    
    def __post_init__(self) -> None:
        self.token = self.token or os.getenv("VAULT_API_KEY")
        if not self.token:
            logger.error("No Vault API token provided")

class SleepAPITools(Toolkit):
    """Toolkit for analyzing sleep data from Vault API with vector search capabilities"""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        use_vector_db: bool = False,  # Make vector DB optional
        vector_db_url: str = "localhost",
        vector_db_port: int = 6333
    ) -> None:
        super().__init__(name="sleep_api_tools")
        self.config = VaultConfig(token=api_key)
        self.use_vector_db = use_vector_db
        
        # Only initialize vector DB if requested
        if use_vector_db:
            self.openai_client = OpenAI()
            self.vector_client = QdrantClient(
                host=vector_db_url,
                port=vector_db_port
            )
            self._init_vector_collection()
        
        # Register methods
        self.register(self.get_sleep_data)
        self.register(self.get_sleep_analysis)
        self.register(self.get_recent_sleep_trends)
        self.register(self.search_sleep_by_date)

    def _init_vector_collection(self) -> None:
        """Initialize vector collection for sleep data"""
        try:
            collections = self.vector_client.get_collections().collections
            if not any(c.name == self.config.collection_name for c in collections):
                self.vector_client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
                )
                logger.info(f"Created vector collection: {self.config.collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize vector collection: {e}")

    def _get_embedding(self, text: str) -> List[float]:
        """Get OpenAI embedding for text"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            return []

    def _store_sleep_entry(self, entry: Dict[str, Any]) -> None:
        """Store a sleep entry in vector database if enabled"""
        if not self.use_vector_db:
            return
            
        try:
            # Create a rich text description for the entry
            entry_text = (
                f"Sleep entry from {entry['date']} - "
                f"Duration: {entry['duration_minutes']} minutes, "
                f"Quality: {entry.get('quality', 'Unknown')}, "
                f"Respiratory Rate: {entry.get('respiratory_rate', 'Unknown')}"
            )
            
            # Get embedding for the entry
            embedding = self._get_embedding(entry_text)
            if not embedding:
                return
                
            # Convert date string to timestamp
            timestamp = datetime.strptime(entry["start_time"], "%Y-%m-%dT%H:%M:%S.%fZ").timestamp()
                
            # Store in Qdrant with timestamp in payload
            self.vector_client.upsert(
                collection_name=self.config.collection_name,
                points=[
                    PointStruct(
                        id=hash(entry["metadata"]["provider_id"]),
                        vector=embedding,
                        payload={
                            "entry": entry,
                            "text_description": entry_text,
                            "timestamp": timestamp,  # Store as numeric timestamp
                            "date": entry["start_time"].split("T")[0]  # Store date for easier filtering
                        }
                    )
                ]
            )
        except Exception as e:
            logger.warning(f"Failed to store entry in vector DB: {e}")

    def search_sleep_patterns(
        self, 
        query: str, 
        limit: int = 5,
        filters: Optional[List[Dict[str, Any]]] = None,
        order_by: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Search for sleep patterns using semantic search with filters
        
        Args:
            query: Natural language query about sleep patterns
            limit: Maximum number of results to return
            filters: List of filter conditions
            order_by: List of sort orders
        """
        try:
            query_embedding = self._get_embedding(query)
            if not query_embedding:
                return self.format_error("Failed to process query")
            
            # Build query conditions
            query_conditions = []
            if filters:
                for f in filters:
                    if "range" in f:
                        query_conditions.append(
                            models.Range(
                                key=f["field"],
                                **f["range"]
                            )
                        )
                    elif "match" in f:
                        query_conditions.append(
                            models.Match(
                                key=f["field"],
                                match=f["match"]
                            )
                        )

            # Perform search with correct filter structure
            results = self.vector_client.search(
                collection_name=self.config.collection_name,
                query_vector=query_embedding,
                query_filter=models.Filter(
                    must=query_conditions
                ) if query_conditions else None,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            # Format results
            matches = []
            for res in results:
                entry = res.payload["entry"]
                matches.append({
                    "date": entry["date"],
                    "similarity_score": res.score,
                    "sleep_data": entry,
                    "description": res.payload["text_description"]
                })
            
            return json.dumps({
                "query": query,
                "matches": matches,
                "total_matches": len(matches),
                "search_timestamp": datetime.now().isoformat(),
                "applied_filters": filters,
                "applied_ordering": order_by
            })
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return self.format_error(f"Search failed: {str(e)}")

    def _parse_timestamp(self, timestamp: str) -> datetime:
        """Parse timestamp string to datetime, handling different formats"""
        formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",  # With microseconds
            "%Y-%m-%dT%H:%M:%SZ",      # Without microseconds
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp, fmt)
            except ValueError:
                continue
                
        raise ValueError(f"Time data '{timestamp}' does not match any expected format")

    def _format_sleep_data_for_display(self, entries: List[Dict[str, Any]]) -> str:
        """Format sleep data for readable display"""
        if not entries:
            return "No sleep data available"
            
        # Sort entries by date
        sorted_entries = sorted(entries, key=lambda x: x["start_time"], reverse=True)
        
        # Create markdown table header
        table = "| Date | Sleep Time | Wake Time | Duration | Quality | Resp Rate |\n"
        table += "|------|------------|-----------|-----------|---------|-----------|\n"
        
        # Add rows with more concise formatting
        for entry in sorted_entries:
            try:
                sleep_time = self._parse_timestamp(entry["start_time"]).strftime("%H:%M")
                wake_time = self._parse_timestamp(entry["end_time"]).strftime("%H:%M")
                
                # Handle None values with default placeholders
                duration = f"{entry['duration_minutes']/60:.1f}h" if entry.get('duration_minutes') else '-'
                quality = str(entry.get('quality', '-'))
                resp_rate = f"{entry['respiratory_rate']:.1f}" if entry.get('respiratory_rate') else '-'
                
                table += f"| {entry['date']} | {sleep_time} | {wake_time} | "
                table += f"{duration} | {quality} | {resp_rate} |\n"
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping malformed entry: {e}")
                continue
            
        # Add concise summary statistics
        summary = "\n### Summary\n"
        
        # Only include valid values in calculations
        durations = [e["duration_minutes"]/60 for e in entries if e.get("duration_minutes")]
        qualities = [e["quality"] for e in entries if e.get("quality") is not None]
        
        if durations:
            summary += f"- Avg Sleep: {sum(durations)/len(durations):.1f}h\n"
            summary += f"- Range: {min(durations):.1f}h - {max(durations):.1f}h\n"
        if qualities:
            summary += f"- Avg Quality: {sum(qualities)/len(qualities):.1f}\n"
            
        # Add entry count
        summary += f"\nShowing {len(entries)} entries"
            
        return table + summary

    def search_sleep_by_date(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        order: str = "desc",
        limit: int = 50,  # Reduced default limit
        max_pages: int = 3  # Reduced max pages
    ) -> str:
        """
        Search sleep entries by date range with pagination
        """
        try:
            all_entries = []
            next_token = None
            pages_fetched = 0
            
            while pages_fetched < max_pages:
                # Build query parameters
                params = {
                    "limit": limit
                }
                if next_token:
                    params["next_token"] = next_token
                
                # Add filter for sorting and date range
                filter_query = [{"field": "created_at", "order": order}]
                
                if start_date:
                    filter_query.append({
                        "field": "created_at",
                        "range": {"gte": f"{start_date}T00:00:00Z"}
                    })
                if end_date:
                    filter_query.append({
                        "field": "created_at",
                        "range": {"lte": f"{end_date}T23:59:59Z"}
                    })
                    
                params["filter"] = json.dumps(filter_query)

                # Make API request
                response = requests.get(
                    f"{self.config.base_url}/entries/by_key/sleep",
                    headers={"Authorization": f"Bearer {self.config.token}"},
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                
                if data.get("status") == "success" and data.get("data"):
                    sleep_entries = data["data"]
                    
                    # Format entries more efficiently
                    for entry in sleep_entries:
                        try:
                            formatted_entry = {
                                "date": entry["start_time"].split("T")[0],
                                "start_time": entry["start_time"],
                                "end_time": entry["end_time"],
                                "duration_minutes": round(float(entry.get("duration", 0)) / 60, 2),
                                "quality": int(entry.get("quality", 0)) if entry.get("quality") is not None else None,
                                "respiratory_rate": round(float(entry.get("respiratory_rate", 0)), 2) if entry.get("respiratory_rate") is not None else None,
                            }
                            all_entries.append(formatted_entry)
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Skipping malformed entry: {e}")
                            continue
                    
                    # Check for next page
                    next_token = data["pagination"].get("next")
                    if not next_token:
                        break
                        
                    pages_fetched += 1
                else:
                    break
            
            # Format the response with the new display method
            if all_entries:
                # Limit the display to most recent entries to avoid token limits
                display_entries = all_entries[:30]  # Show only last 30 entries in table
                formatted_display = self._format_sleep_data_for_display(display_entries)
                
                return json.dumps({
                    "display": formatted_display,
                    "summary": {
                        "total_entries_found": len(all_entries),
                        "total_available": data["pagination"]["total"],
                        "pages_fetched": pages_fetched + 1,
                        "has_more": bool(next_token),
                        "date_range": {
                            "start": start_date,
                            "end": end_date
                        }
                    }
                })
            else:
                return self.format_error("No sleep data available for the specified date range")
                
        except Exception as e:
            logger.error(f"Date search failed: {str(e)}")
            return self.format_error(f"Date search failed: {str(e)}")

    def get_sleep_data(self, days: int = 7, next_token: Optional[str] = None) -> str:
        """
        Fetch sleep data for the specified number of days
        
        Args:
            days: Number of days of sleep data to fetch
            next_token: Pagination token for next page
        """
        if not self.config.token:
            return self.format_error("No API token configured")
            
        try:
            # Build query parameters
            params = {
                "limit": days
            }
            if next_token:
                params["next_token"] = next_token
                
            # Add sorting by created_at desc to get most recent entries
            filter_query = [{"field": "created_at", "order": "desc"}]
            params["filter"] = json.dumps(filter_query)
            
            # Make API request
            response = requests.get(
                f"{self.config.base_url}/entries/by_key/sleep",
                headers={"Authorization": f"Bearer {self.config.token}"},
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "success" and data.get("data"):
                sleep_entries = data["data"]
                
                # Format entries
                formatted_entries = []
                for entry in sleep_entries:
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
                        
                        # Store in vector DB if available
                        self._store_sleep_entry(formatted_entry)
                        
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Skipping malformed entry: {e}")
                        continue
                
                return json.dumps({
                    "recent_sleep_data": formatted_entries,
                    "days_analyzed": len(formatted_entries),
                    "total_available": data["pagination"]["total"],
                    "next_token": data["pagination"].get("next"),
                    "metadata": {
                        "fragment_type": "sleep",
                        "timestamp": datetime.now().isoformat()
                    }
                })
            else:
                return self.format_error("No sleep data available or invalid response")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return self.format_error(f"API request failed: {str(e)}")
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