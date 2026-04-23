from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()


class KnowledgeGraph:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
        )

    def close(self):
        self.driver.close()

    def clear_graph(self):
        with self.driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")

    def build_graph(self, data: dict):
        """Creates Person/Location/Event nodes + relationships:
           (Person)-[:PARTICIPATED_IN]->(Event)
           (Event)-[:OCCURRED_IN]->(Location)
        """
        d = data["detailed"]

        with self.driver.session() as s:

            # Persons
            for p in d["persons"]:
                s.run(
                    """MERGE (n:Person {name: $name})
                       SET n.role = $role, n.confidence = $conf""",
                    name=p["name"], role=p.get("role", ""),
                    conf=p.get("confidence", 0.9),
                )

            # Locations
            for l in d["locations"]:
                s.run(
                    """MERGE (n:Location {name: $name})
                       SET n.type = $type, n.confidence = $conf""",
                    name=l["name"], type=l.get("type", ""),
                    conf=l.get("confidence", 0.9),
                )

            # Events + relationships
            for e in d["events"]:
                s.run(
                    """MERGE (ev:Event {name: $name})
                       SET ev.year = $year, ev.confidence = $conf""",
                    name=e["event"], year=e.get("year"),
                    conf=e.get("confidence", 0.9),
                )

                if e.get("location"):
                    s.run(
                        """MATCH (ev:Event {name: $ename})
                           MERGE (loc:Location {name: $lname})
                           MERGE (ev)-[:OCCURRED_IN]->(loc)""",
                        ename=e["event"], lname=e["location"],
                    )

                for person in e.get("participants", []):
                    s.run(
                        """MATCH (ev:Event {name: $ename})
                           MERGE (p:Person {name: $pname})
                           MERGE (p)-[:PARTICIPATED_IN]->(ev)""",
                        ename=e["event"], pname=person,
                    )