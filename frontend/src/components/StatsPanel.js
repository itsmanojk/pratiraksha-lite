import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';

const PanelContainer = styled(motion.div)`
  background: rgba(0, 124, 145, 0.1);
  backdrop-filter: blur(15px);
  border-radius: 20px;
  border: 1px solid rgba(0, 124, 145, 0.3);
  padding: 2rem;
  height: fit-content;
`;

const PanelTitle = styled.h2`
  font-size: 1.5rem;
  font-weight: 600;
  margin-bottom: 1.5rem;
  color: #004d40;
`;

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
`;

const StatCard = styled(motion.div)`
  background: rgba(0, 124, 145, 0.15);
  border-radius: 12px;
  padding: 1.5rem;
  text-align: center;
  border: 1px solid rgba(0, 124, 145, 0.3);
  transition: all 0.3s ease;

  &:hover {
    transform: translateY(-2px);
    background: rgba(0, 124, 145, 0.25);
  }
`;

const StatNumber = styled.div`
  font-size: 2.5rem;
  font-weight: 700;
  color: #004d40;
  margin-bottom: 0.5rem;
  font-family: 'JetBrains Mono', monospace;
`;

const StatLabel = styled.div`
  font-size: 0.9rem;
  color: #006064;
  font-weight: 500;
`;

const UptimeCard = styled(StatCard)`
  grid-column: 1 / -1;
  margin-top: 1rem;
`;

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.1 },
  },
};

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0 },
};

const StatsPanel = ({ stats }) => {
  return (
    <PanelContainer variants={container} initial="hidden" animate="show" whileHover={{ scale: 1.02 }} transition={{ type: "spring", stiffness: 300 }}>
      <PanelTitle>📊 Live Statistics</PanelTitle>
      <StatsGrid>
        <StatCard variants={item}>
          <StatNumber>{stats.total_flows.toLocaleString()}</StatNumber>
          <StatLabel>Total Flows</StatLabel>
        </StatCard>
        <StatCard variants={item}>
          <StatNumber>{stats.threats_detected.toLocaleString()}</StatNumber>
          <StatLabel>Threats Detected</StatLabel>
        </StatCard>
        <StatCard variants={item}>
          <StatNumber>{stats.threats_blocked.toLocaleString()}</StatNumber>
          <StatLabel>Threats Blocked</StatLabel>
        </StatCard>
        <StatCard variants={item}>
          <StatNumber>{stats.detection_rate}</StatNumber>
          <StatLabel>Detection Rate</StatLabel>
        </StatCard>
      </StatsGrid>

      <UptimeCard variants={item}>
        <StatNumber>{stats.uptime}</StatNumber>
        <StatLabel>System Uptime</StatLabel>
      </UptimeCard>
    </PanelContainer>
  );
};

export default StatsPanel;
