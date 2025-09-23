from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
import datetime

Base = declarative_base()

class ThreatLog(Base):
    __tablename__ = 'threat_logs'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    source_ip = Column(String)
    dest_ip = Column(String)
    threat_type = Column(String)
    confidence = Column(Float)
    status = Column(String)

engine = create_engine('sqlite:///database.db', connect_args={'check_same_thread': False})
Session = sessionmaker(bind=engine)

def init_db():
    Base.metadata.create_all(engine)

def log_threat(threat):
    session = Session()
    # Accept both datetime and string, convert string to datetime if needed
    ts = threat.get('timestamp')
    if isinstance(ts, str):
        try:
            ts = datetime.datetime.fromisoformat(ts)
        except Exception:
            ts = datetime.datetime.utcnow()
    threat_log = ThreatLog(
        timestamp=ts,
        source_ip=threat.get('source_ip'),
        dest_ip=threat.get('dest_ip'),
        threat_type=threat.get('threat_type'),
        confidence=threat.get('confidence'),
        status=threat.get('status')
    )
    session.add(threat_log)
    session.commit()
    session.close()

def get_stats():
    session = Session()
    total = session.query(ThreatLog).count()
    blocked = session.query(ThreatLog).filter(ThreatLog.status == 'BLOCKED').count()
    detected = session.query(ThreatLog).count()
    session.close()
    return {
        "total_flows": 0,
        "threats_detected": detected,
        "threats_blocked": blocked,
        "benign_flows": 0,
        "detection_rate": "N/A",
        "uptime": "N/A"
    }
