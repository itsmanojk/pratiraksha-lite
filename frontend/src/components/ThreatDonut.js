import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';

const COLORS = ['#f54e42', '#f5a623', '#c471ed', "#0088FE", "#FFBB28", "#00C49F", "#FF8042"];

function ThreatDonut({ data }) {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <PieChart>
        <Pie data={data} dataKey="value" innerRadius={60} outerRadius={90} fill="#82ca9d" label>
          {data.map((entry, index) => <Cell key={index} fill={COLORS[index % COLORS.length]}/>)}
        </Pie>
        <Tooltip />
        <Legend />
      </PieChart>
    </ResponsiveContainer>
  );
}
export default ThreatDonut;
