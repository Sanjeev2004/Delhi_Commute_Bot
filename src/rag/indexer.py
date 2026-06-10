"""
Transit data indexer — builds FAISS vector indices from route data.

Uses the ``sentence-transformers/all-MiniLM-L6-v2`` embedding model
via the LangChain ``HuggingFaceEmbeddings`` wrapper and stores indices
through the LangChain ``FAISS`` vector store.

Usage::

    indexer = TransitIndexer()
    indexer.build_all()  # writes indices to data/indices/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from loguru import logger

from src.config import settings


_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class TransitIndexer:
    """Creates and persists FAISS indices from local transit data files."""

    def __init__(self) -> None:
        self._embeddings = HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL)
        self._index_dir = Path(settings.faiss_index_path)
        self._index_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _load_json(path: str | Path) -> Any:
        p = Path(path)
        if not p.exists():
            logger.warning("Data file not found: {}", p)
            return None
        with p.open(encoding="utf-8") as fh:
            return json.load(fh)

    def _save_index(self, store: FAISS, name: str) -> Path:
        dest = self._index_dir / name
        store.save_local(str(dest))
        logger.info("Saved FAISS index '{}' -> {}", name, dest)
        return dest

    def _build_bus_documents(self) -> list[Document]:
        docs: list[Document] = []
        gtfs_path = Path(settings.active_gtfs_path)
        routes_file = gtfs_path / "routes.txt"

        if routes_file.exists():
            import csv
            with routes_file.open(encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    route_id = row.get("route_id", "")
                    name = row.get("route_long_name") or row.get("route_short_name", "")
                    desc = row.get("route_desc", "")
                    text = f"DTC Bus Route {route_id}: {name}. {desc}"
                    docs.append(Document(page_content=text, metadata={"type": "bus", "route_id": route_id, "name": name}))
            logger.info("Built {} bus documents from GTFS", len(docs))
        else:
            sample_routes = [
                {"id": "473", "name": "Kashmere Gate -> Laxmi Nagar", "via": "Shastri Park, Jheel", "stops": "Kashmere Gate ISBT, Shastri Park, Seelampur, Shahdara, Laxmi Nagar"},
                {"id": "764", "name": "Mehrauli -> ISBT Kashmere Gate", "via": "Saket, AIIMS, INA, Central Secretariat", "stops": "Mehrauli, Saket, Malviya Nagar, AIIMS, INA, RK Ashram, Kashmere Gate"},
                {"id": "604", "name": "Dwarka Sector 23 -> Old Delhi", "via": "Janakpuri, Rajouri Garden, Karol Bagh", "stops": "Dwarka Sec 23, Dwarka Mor, Uttam Nagar, Janakpuri, Rajouri Garden, Karol Bagh, Old Delhi"},
                {"id": "340", "name": "Noida Sector 37 -> Mandi House", "via": "Mayur Vihar, Nizamuddin", "stops": "Noida Sec 37, Mayur Vihar Phase 1, Sarai Kale Khan, Nizamuddin, India Gate, Mandi House"},
                {"id": "981", "name": "Mundka -> Connaught Place", "via": "Tikri Border, Nangloi, Punjabi Bagh", "stops": "Mundka, Nangloi, Rajdhani Park, Punjabi Bagh, Patel Nagar, Connaught Place"},
                {"id": "534", "name": "Nehru Place -> Shakti Nagar", "via": "Defence Colony, Lodi Road", "stops": "Nehru Place, Kalkaji, Defence Colony, Lodi Road, India Gate, Civil Lines, Shakti Nagar"},
                {"id": "729", "name": "Dilshad Garden -> Badarpur Border", "via": "Anand Vihar, Laxmi Nagar, Ashram", "stops": "Dilshad Garden, Anand Vihar, Laxmi Nagar, Ashram, Sarita Vihar, Badarpur"},
                {"id": "427", "name": "Rohini Sector 22 -> Central Secretariat", "via": "Pitampura, Netaji Subhash Place, Wazirpur", "stops": "Rohini Sec 22, Pitampura, NSP, Wazirpur, Patel Nagar, Connaught Place, Central Secretariat"},
            ]
            for route in sample_routes:
                text = f"DTC Bus Route {route['id']}: {route['name']}. Via {route['via']}. Stops: {route['stops']}."
                docs.append(Document(page_content=text, metadata={"type": "bus", "route_id": route["id"], "name": route["name"]}))
            logger.info("Built {} bus documents from sample data", len(docs))
        return docs

    def build_bus_index(self) -> FAISS | None:
        docs = self._build_bus_documents()
        if not docs:
            logger.warning("No bus documents to index.")
            return None
        store = FAISS.from_documents(docs, self._embeddings)
        self._save_index(store, "bus_index")
        return store

    def _build_metro_documents(self) -> list[Document]:
        docs: list[Document] = []
        metro_lines = [
            {"line": "Red Line", "color": "Red", "from": "Shaheed Sthal", "to": "Rithala", "stations": ["Shaheed Sthal", "Hindon River", "Arthala", "Mohan Nagar", "Shyam Park", "Dilshad Garden", "Jhilmil", "Mansarovar Park", "Shahdara", "Welcome", "Seelampur", "Shastri Park", "Kashmere Gate", "Tis Hazari", "Pul Bangash", "Pratap Nagar", "Shastri Nagar", "Inderlok", "Kanhaiya Nagar", "Keshav Puram", "Netaji Subhash Place", "Kohat Enclave", "Pitampura", "Rohini East", "Rohini West", "Rithala"]},
            {"line": "Blue Line", "color": "Blue", "from": "Dwarka Sector 21", "to": "Vaishali", "stations": ["Dwarka Sector 21", "Dwarka Sector 8", "Dwarka Sector 9", "Dwarka Sector 10", "Dwarka Sector 11", "Dwarka Sector 12", "Dwarka Sector 13", "Dwarka Sector 14", "Dwarka", "Dwarka Mor", "Nawada", "Uttam Nagar East", "Uttam Nagar West", "Janakpuri West", "Janakpuri East", "Tilak Nagar", "Subhash Nagar", "Tagore Garden", "Rajouri Garden", "Ramesh Nagar", "Moti Nagar", "Kirti Nagar", "Shadipur", "Patel Nagar", "Rajendra Place", "Karol Bagh", "Jhandewalan", "RK Ashram Marg", "Rajiv Chowk", "Barakhamba Road", "Mandi House", "Pragati Maidan", "Indraprastha", "Yamuna Bank", "Laxmi Nagar", "Nirman Vihar", "Preet Vihar", "Karkarduma", "Anand Vihar ISBT", "Kaushambi", "Vaishali"]},
            {"line": "Yellow Line", "color": "Yellow", "from": "Samaypur Badli", "to": "HUDA City Centre", "stations": ["Samaypur Badli", "Rohini Sector 18-19", "Haiderpur Badli Mor", "Jahangirpuri", "Adarsh Nagar", "Azadpur", "Model Town", "GTB Nagar", "Vishwavidyalaya", "Vidhan Sabha", "Civil Lines", "Kashmere Gate", "Chandni Chowk", "Chawri Bazar", "New Delhi", "Rajiv Chowk", "Patel Chowk", "Central Secretariat", "Udyog Bhawan", "Lok Kalyan Marg", "Jorbagh", "INA", "AIIMS", "Green Park", "Hauz Khas", "Malviya Nagar", "Saket", "Qutab Minar", "Chhattarpur", "Sultanpur", "Ghitorni", "Arjan Garh", "Guru Dronacharya", "Sikandarpur", "MG Road", "IFFCO Chowk", "HUDA City Centre"]},
            {"line": "Violet Line", "color": "Violet", "from": "Kashmere Gate", "to": "Raja Nahar Singh", "stations": ["Kashmere Gate", "Lal Quila", "Jama Masjid", "Delhi Gate", "ITO", "Mandi House", "Janpath", "Central Secretariat", "Khan Market", "JLN Stadium", "Jangpura", "Lajpat Nagar", "Moolchand", "Kailash Colony", "Nehru Place", "Kalkaji Mandir", "Govind Puri", "Okhla", "Jasola Apollo", "Sarita Vihar", "Mohan Estate", "Tughlakabad", "Badarpur Border"]},
            {"line": "Green Line", "color": "Green", "from": "Inderlok", "to": "Brigadier Hoshiar Singh", "stations": ["Inderlok", "Ashok Park Main", "Punjabi Bagh", "Shivaji Park", "Madipur", "Paschim Vihar East", "Paschim Vihar West", "Peera Garhi", "Udyog Nagar", "Surajmal Stadium", "Nangloi", "Rajdhani Park", "Mundka", "Tikri Border", "Bahadurgarh City", "Brigadier Hoshiar Singh"]},
            {"line": "Magenta Line", "color": "Magenta", "from": "Botanical Garden", "to": "Janakpuri West", "stations": ["Botanical Garden", "Okhla Bird Sanctuary", "Kalindi Kunj", "Jasola Vihar Shaheen Bagh", "Okhla NSIC", "Sukhdev Vihar", "Jamia Millia Islamia", "Okhla Vihar", "Jasola Apollo", "Sarita Vihar", "Hauz Khas", "IIT Delhi", "R K Puram", "Munirka", "Vasant Vihar", "Shankar Vihar", "Terminal 1 IGI Airport", "Palam", "Dashrath Puri", "Dabri Mor", "Janakpuri West"]},
        ]

        for line in metro_lines:
            stations_str = ", ".join(line["stations"])
            text = f"Delhi Metro {line['line']} ({line['color']}): From {line['from']} to {line['to']}. Stations: {stations_str}."
            docs.append(Document(page_content=text, metadata={"type": "metro", "line": line["line"], "color": line["color"], "station_count": len(line["stations"])}))
            for idx, station in enumerate(line["stations"]):
                neighbors = []
                if idx > 0:
                    neighbors.append(line["stations"][idx - 1])
                if idx < len(line["stations"]) - 1:
                    neighbors.append(line["stations"][idx + 1])
                station_text = f"{station} metro station on the {line['line']} ({line['color']}). Adjacent stations: {', '.join(neighbors)}."
                docs.append(Document(page_content=station_text, metadata={"type": "metro_station", "station": station, "line": line["line"], "color": line["color"], "position": idx}))

        logger.info("Built {} metro documents (lines + stations)", len(docs))
        return docs

    def build_metro_index(self) -> FAISS | None:
        docs = self._build_metro_documents()
        if not docs:
            logger.warning("No metro documents to index.")
            return None
        store = FAISS.from_documents(docs, self._embeddings)
        self._save_index(store, "metro_index")
        return store

    def _build_shared_auto_documents(self) -> list[Document]:
        data = self._load_json(settings.shared_auto_data_path + "/routes.json")
        if not data:
            return []
        docs: list[Document] = []
        for route in data.get("routes", []):
            via_str = ", ".join(route.get("via", []))
            text = f"Shared auto route {route['id']}: {route['name']}. From {route['from']} to {route['to']}. Via: {via_str}. Fare: Rs.{route['fare_inr']}. Type: {route.get('type', 'shared_auto')}. Frequency: {route.get('frequency', 'varies')}. Hours: {route.get('operating_hours', 'N/A')}."
            docs.append(Document(page_content=text, metadata={"type": "shared_auto", "route_id": route["id"], "from": route["from"], "to": route["to"], "fare_inr": route["fare_inr"]}))
        logger.info("Built {} shared-auto documents", len(docs))
        return docs

    def _build_auto_documents(self) -> list[Document]:
        data = self._load_json(settings.auto_data_path + "/fare_chart.json")
        if not data:
            return []
        docs: list[Document] = []
        fare = data.get("fare_structure", {})
        docs.append(Document(page_content=f"Delhi auto-rickshaw fare structure: Base fare Rs.{fare.get('base_fare_inr', 30)} for first {fare.get('base_distance_km', 1.5)} km. Rs.{fare.get('per_km_rate_inr', 11)} per km after that. Night surcharge {fare.get('night_surcharge_pct', 25)}% between {fare.get('night_start_hour', 23)}:00 and {fare.get('night_end_hour', 5)}:00.", metadata={"type": "auto_fare", "subtype": "structure"}))
        for route in data.get("common_routes", []):
            text = f"Auto from {route['from']} to {route['to']}: Distance {route['distance_km']} km. Meter fare Rs.{route['typical_meter_fare_inr']}. Typical asking fare Rs.{route['typical_asking_fare_inr']}."
            docs.append(Document(page_content=text, metadata={"type": "auto_fare", "subtype": "route", "from": route["from"], "to": route["to"], "distance_km": route["distance_km"]}))
        logger.info("Built {} auto-fare documents", len(docs))
        return docs

    def build_shared_auto_index(self) -> FAISS | None:
        docs = self._build_shared_auto_documents()
        if not docs:
            return None
        store = FAISS.from_documents(docs, self._embeddings)
        self._save_index(store, "shared_auto_index")
        return store

    def build_auto_index(self) -> FAISS | None:
        docs = self._build_auto_documents()
        if not docs:
            return None
        store = FAISS.from_documents(docs, self._embeddings)
        self._save_index(store, "auto_index")
        return store

    def build_all(self) -> dict[str, FAISS | None]:
        """Build all indices and persist to data/indices/."""
        logger.info("Building all FAISS indices ...")
        results = {"bus": self.build_bus_index(), "metro": self.build_metro_index(), "shared_auto": self.build_shared_auto_index(), "auto": self.build_auto_index()}
        logger.info("Index build complete. Built: {}", [k for k, v in results.items() if v is not None])
        return results


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    TransitIndexer().build_all()
