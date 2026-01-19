import JobView from 'src/components/dashboard/JobView';

const Job = () => {

  return (
    <div className="grid grid-cols-12 gap-30">
      <div className="lg:col-span-10 col-span-12">
        <JobView />
      </div>
    </div>
  );
};

export default Job;
