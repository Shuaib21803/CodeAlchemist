import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Loader2, Upload } from "lucide-react";

const CodeAlchemist = () => {
  const [weightType, setWeightType] = useState("");
  const [jupyterFile, setJupyterFile] = useState(null);
  const [modelWeights, setModelWeights] = useState("");
  const [modelDetails, setModelDetails] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleJupyterUpload = (e) => {
    const file = e.target.files[0];
    if (file && file.name.endsWith('.ipynb')) {
      setJupyterFile(file);
    } else {
      setError("Please upload a valid Jupyter notebook file (.ipynb)");
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (!weightType || !jupyterFile || !modelWeights || !modelDetails) {
      setError("Please fill in all required fields");
      return;
    }

    setLoading(true);
    setError(null);

    // Simulate processing
    setTimeout(() => {
      setLoading(false);
      // Add success handling here
    }, 2000);
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <Card className="max-w-2xl mx-auto">
        <CardHeader>
          <CardTitle className="text-3xl font-bold text-center">
            CodeAlchemist Workflow
          </CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Step 1: Select weights type */}
            <div className="space-y-2">
              <label className="text-sm font-medium">
                Select Weights Type
              </label>
              <Select
                value={weightType}
                onValueChange={setWeightType}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Choose weights type..." />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="pytorch">PyTorch Weights</SelectItem>
                  <SelectItem value="tensorflow">TensorFlow Weights</SelectItem>
                  <SelectItem value="onnx">ONNX Weights</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Step 2: User Inputs Section */}
            <div className="border rounded-lg p-4 space-y-4">
              <h3 className="font-medium">User Inputs</h3>
              
              {/* Jupyter File Upload */}
              <div className="space-y-2">
                <label className="text-sm font-medium">
                  Jupyter Notebook File
                </label>
                <div className="flex items-center space-x-2">
                  <Input
                    type="file"
                    accept=".ipynb"
                    onChange={handleJupyterUpload}
                    className="flex-1"
                  />
                  <Upload className="h-5 w-5 text-gray-400" />
                </div>
              </div>

              {/* Model Weights Input */}
              <div className="space-y-2">
                <label className="text-sm font-medium">
                  Model Weights
                </label>
                <Textarea
                  placeholder="Enter model weights configuration..."
                  value={modelWeights}
                  onChange={(e) => setModelWeights(e.target.value)}
                  className="min-h-[100px]"
                />
              </div>

              {/* Model Details Input */}
              <div className="space-y-2">
                <label className="text-sm font-medium">
                  Model Details
                </label>
                <Textarea
                  placeholder="Enter model details and parameters..."
                  value={modelDetails}
                  onChange={(e) => setModelDetails(e.target.value)}
                  className="min-h-[100px]"
                />
              </div>
            </div>

            {/* Submit Button */}
            <Button 
              type="submit" 
              className="w-full"
              disabled={loading}
            >
              {loading ? (
                <div className="flex items-center justify-center">
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Processing...
                </div>
              ) : (
                "Transform with CodeAlchemist"
              )}
            </Button>
          </form>

          {error && (
            <Alert variant="destructive" className="mt-4">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default CodeAlchemist;
