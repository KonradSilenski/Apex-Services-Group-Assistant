
import FullLogo from "src/layouts/full/shared/logo/FullLogo";
import AuthLogin from "../authforms/AuthLogin";

const gradientStyle = {
  background: "bg-lightgray",
  backgroundSize: "400% 400%",
  height: "100vh",
};

const Login = () => {
  return (
    <div style={gradientStyle} className="relative overflow-hidden h-screen">
      <div className="flex h-full justify-center items-center px-4">
        <div className="rounded-xl shadow-md bg-blue-500 dark:bg-blue-700 p-6 w-full md:w-96 border-none">
          <div className="flex flex-col gap-2 p-0 w-full">
            <div className="mx-auto">
              <FullLogo />
            </div>
            <p className="text-sm font-bold text-center text-white my-3">Sign In on Apex Assistant</p>
            <AuthLogin />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;
