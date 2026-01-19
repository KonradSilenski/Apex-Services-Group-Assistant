import ProductRevenue from 'src/components/dashboard/ProductRevenue';

const Dashboard = () => {
  return (
    <div className="grid grid-cols-12 gap-30">
      <div className="lg:col-span-10 col-span-12">
        <ProductRevenue />
      </div>
    </div>
  );
};

export default Dashboard;
