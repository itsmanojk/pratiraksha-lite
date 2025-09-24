import React from 'react';
import styled from 'styled-components';

const Grid = styled.div`
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 2rem;
  margin-bottom: 2rem;
`;

const Card = styled.div`
  background: rgba(23,25,48,0.85);
  border-radius: 18px;
  padding: 2rem 1.5rem;
  box-shadow: 0 8px 20px rgba(5,10,24,0.10);
  color: #fff;
  text-align: left;
`;

const Stat = styled.div`
  font-size: 2.4rem;
  font-weight: bold;
`;

const Label = styled.div`
  color: #cfcfff;
  margin-top: 0.5rem;
  font-size: 1rem;
  font-weight: 600;
`;

function StatsCards({ stats }) {
  return (
    <Grid>
      <Card>
        <Stat>{stats.total_flows?.toLocaleString() || '--'}</Stat>
        <Label>Packets Processed</Label>
      </Card>
      <Card>
        <Stat>{stats.threats_detected?.toLocaleString() || '--'}</Stat>
        <Label>Threats Detected</Label>
      </Card>
      <Card>
        <Stat>{stats.threats_blocked?.toLocaleString() || '--'}</Stat>
        <Label>Threats Blocked</Label>
      </Card>
      <Card>
        <Stat>{stats.detection_rate || '--'}</Stat>
        <Label>Detection Accuracy</Label>
      </Card>
    </Grid>
  );
}

export default StatsCards;
