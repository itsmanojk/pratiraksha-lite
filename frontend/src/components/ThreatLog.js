import React from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';

const LogContainer = styled(motion.div)`
  background: rgba(0, 124, 145, 0.1);
  backdrop-filter: blur(15px);
  border-radius: 20px;
  border: 1px solid rgba(0, 124, 145, 0.3);
  padding: 2rem;
`;

const LogTitle = styled.h2`
  font-size: 1.5rem;
  font-weight: 600;
  margin-bottom: 1.5rem;
  color: #004d40;
`;

const LogContent = styled.div`
  max-height: 500px;
  overflow-y: auto;
  padding-right: 0.5rem;
`;

const EmptyState = styled.div`
  text-align: center;
  padding: 3rem;
  color: #004d40aa;
  font-size: 1rem;
`;

const ThreatItem = styled(motion.div)`
  background: ${props => {
    switch (props.status) {
      case 'BLOCKED': return 'rgba(239, 68, 68, 0.1)';
      case 'MONITORED': return 'rgba(245, 158, 11, 0.1)';
      default: return 'rgba(156, 163, 175, 0.1)';
    }
  }};
  border-left: 4px solid ${props => {
    switch (props.status) {
      case 'BLOCKED': return '#ef4444';
      case 'MONITORED': return '#f59e0b';
      default: return '#9ca3af';
    }
  }};
  border-radius: 8px;
  padding: 1rem;
  margin-bottom: 0.75rem;
  border: 1px solid ${props => {
    switch (props.status) {
      case 'BLOCKED': return 'rgba(239, 68, 68, 0.3)';
      case 'MONITORED': return 'rgba(245, 158, 11, 0.3)';
      default: return 'rgba(156, 163, 175, 0.3)';
    }
  }};
`;

const ThreatHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
`;

const ThreatTime = styled.span`
  font-weight: 600;
  color: #004d40;
  font-family: 'JetBrains Mono', monospace;
`;

const ThreatType = styled.span`
  font-weight: 700;
  color: #ef4444;
  font-size: 1.1rem;
`;

const ThreatDetails = styled.div`
  font-size: 0.9rem;
  color: #006064;
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
`;

const ThreatDetail = styled.span`
  display: flex;
  align-items: center;
  gap: 0.25rem;
`;

const StatusBadge = styled.span`
  padding: 0.25rem 0.5rem;
  border-radius: 6px;
  font-size: 0.75rem;
  font-weight: 600;
  background: ${props => props.status === 'BLOCKED' ? 'rgba(239, 68, 68, 0.2)' : 'rgba(245, 158, 11, 0.2)'};
  color: ${props => props.status === 'BLOCKED' ? '#ef4444' : '#f59e0b'};
  border: 1px solid ${props => props.status === 'BLOCKED' ? '#ef4444' : '#f59e0b'};
`;

const ThreatLog = ({ threats }) => {
  return (
    <LogContainer
      initial={{ opacity: 0, y: 50 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.4, duration: 0.8 }}
    >
      <LogTitle>🚨 Real-Time Threat Detection Log</LogTitle>

      <LogContent>
        {threats.length === 0 ? (
          <EmptyState>
            🔍 Monitoring network traffic... Threats will appear here in real-time.
          </EmptyState>
        ) : (
          <AnimatePresence>
            {threats.map((threat, index) => (
              <ThreatItem
                key={`${threat.timestamp}-${index}`}
                status={threat.status}
                initial={{ opacity: 0, x: -50, scale: 0.9 }}
                animate={{ opacity: 1, x: 0, scale: 1 }}
                exit={{ opacity: 0, x: 50, scale: 0.9 }}
                transition={{ type: "spring", stiffness: 300, damping: 30 }}
                layout
              >
                <ThreatHeader>
                  <ThreatTime>{threat.timestamp}</ThreatTime>
                  <ThreatType>{threat.threat_type} detected</ThreatType>
                </ThreatHeader>

                <ThreatDetails>
                  <ThreatDetail>
                    📍 {threat.source_ip} → {threat.dest_ip}
                  </ThreatDetail>
                  <ThreatDetail>
                    🎯 Confidence: {threat.confidence}
                  </ThreatDetail>
                  <ThreatDetail>
                    Status: <StatusBadge status={threat.status}>{threat.status}</StatusBadge>
                  </ThreatDetail>
                </ThreatDetails>
              </ThreatItem>
            ))}
          </AnimatePresence>
        )}
      </LogContent>
    </LogContainer>
  );
};

export default ThreatLog;
