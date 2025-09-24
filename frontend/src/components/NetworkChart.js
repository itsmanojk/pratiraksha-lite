import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

function NetworkChart({ data }) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#353556" />
        <XAxis dataKey="time" tick={{ fill: "#fff" }} />
        <YAxis tick={{ fill: "#fff" }} />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="packets" stroke="#00ff7f" />
        <Line type="monotone" dataKey="threats" stroke="#f54e42" />
      </LineChart>
    </ResponsiveContainer>
  );
}
export default NetworkChart;
