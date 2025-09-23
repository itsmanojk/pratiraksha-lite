import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';

const ModelContainer = styled(motion.div)`
  background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(16, 185, 129, 0.1) 100%);
  backdrop-filter: blur(20px);
  border-radius: 20px;
  border: 1px solid rgba(34, 197, 94, 0.3);
  padding: 2rem;
  height: fit-content;
`;

const ModelTitle = styled.h2`
  font-size: 1.5rem;
  font-weight: 600;
  margin-bottom: 1.5rem;
  color: #f8fafc;
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const ModelInfo = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1rem;
`;

const InfoRow = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.75rem 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);

  &:last-child {
    border-bottom: none;
  }
`;

const InfoLabel = styled.span`
  font-weight: 500;
  color: #94a3b8;
`;

const InfoValue = styled.span`
  font-weight: 600;
  color: #f8fafc;
  font-family: 'JetBrains Mono', monospace;
`;

const StatusBadge = styled(motion.span)`
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-size: 0.8rem;
  font-weight: 600;
  background: ${props => props.active ? 'rgba(34, 197, 94, 0.2)' : 'rgba(239, 68, 68, 0.2)'};
  color: ${props => props.active ? '#22c55e' : '#ef4444'};
  border: 1px solid ${props => props.active ? '#22c55e' : '#ef4444'};
`;

const AccuracyBar = styled.div`
  width: 100%;
  height: 6px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 3px;
  overflow: hidden;
`;

const AccuracyFill = styled(motion.div)`
  height: 100%;
  background: linear-gradient(90deg, #22c55e 0%, #16a34a 100%);
  border-radius: 3px;
`;

const ModelStatus = ({ modelInfo, connected }) => {
  const accuracy = 79.2;
  
  return (
    <ModelContainer
      initial={{ opacity: 0, x: 50 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: 0.2, duration: 0.8 }}
      whileHover={{ scale: 1.02 }}
    >
      <ModelTitle>🧠 AI Model Status</ModelTitle>
      
      <ModelInfo>
        <InfoRow>
          <InfoLabel>Model Type:</InfoLabel>
          <InfoValue>Graph Convolutional Network</InfoValue>
        </InfoRow>
        
        <InfoRow>
          <InfoLabel>Status:</InfoLabel>
          <StatusBadge 
            active={connected}
            animate={{ scale: connected ? [1, 1.05, 1] : 1 }}
            transition={{ repeat: connected ? Infinity : 0, duration: 2 }}
          >
            {connected ? 'Active & Monitoring' : 'Disconnected'}
          </StatusBadge>
        </InfoRow>
        
        <InfoRow>
          <InfoLabel>Parameters:</InfoLabel>
          <InfoValue>{modelInfo?.parameters?.toLocaleString() || '22,309'}</InfoValue>
        </InfoRow>
        
        <InfoRow>
          <InfoLabel>Classes:</InfoLabel>
          <InfoValue>{modelInfo?.classes || '5'} Threat Types</InfoValue>
        </InfoRow>
        
        <InfoRow>
          <InfoLabel>Accuracy:</InfoLabel>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '0.5rem', flex: 1 }}>
            <InfoValue>{accuracy}%</InfoValue>
            <AccuracyBar>
              <AccuracyFill
                initial={{ width: 0 }}
                animate={{ width: `${accuracy}%` }}
                transition={{ delay: 1, duration: 2, ease: "easeOut" }}
              />
            </AccuracyBar>
          </div>
        </InfoRow>
      </ModelInfo>
    </ModelContainer>
  );
};

export default ModelStatus;
