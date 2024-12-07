"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Input } from "../ui/input";
import { Button } from "../ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import { Textarea } from "../ui/textarea";
import { Alert, AlertDescription } from "../ui/alert";
import { Loader2, Upload } from "lucide-react";

import "../../../styles.css";

const CodeAlchemist = () => {
  const [weightType, setWeightType] = useState<string | undefined>();
  const [jupyterFile, setJupyterFile] = useState<File | null>(null);
  const [modelWeights, setModelWeights] = useState("");
  const [modelDetails, setModelDetails] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleJupyterUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.name.endsWith(".ipynb")) {
      setJupyterFile(file);
      setError(null); // Clear error if a valid file is uploaded
    } else {
      setError("Please upload a valid Jupyter notebook file (.ipynb)");
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
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
      alert("Successfully transformed with CodeAlchemist!");
      // Reset form or handle success logic
    }, 2000);
  };

  return (
    <div className="parent-container">
      <Card className="card">
        <CardHeader>
          <CardTitle className="head">CodeAlchemist</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="card-content">
            {/* Step 1: Select weights type */}
            <div className="box">
              <label className="label">Select Weights Type</label>
              <div className="small-box">
                <Select
                  value={weightType}
                  onValueChange={setWeightType}
                >
                  <SelectTrigger className="select-trigger">
                    <SelectValue placeholder="Choose weights type..." />
                  </SelectTrigger>
                  <SelectContent className="select-content">
                    <SelectItem value="pytorch" className="select-item">PyTorch Weights</SelectItem>
                    <SelectItem value="tensorflow" className="select-item">TensorFlow Weights</SelectItem>
                    <SelectItem value="onnx" className="select-item">ONNX Weights</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Step 2: User Inputs Section */}
            <div className="column">
              {/* Jupyter File Upload */}
              <div className="box">
                <label className="label">Jupyter Notebook File</label>
                <div className="flex items-center space-x-2">
                  <Input
                    type="file"
                    accept=".ipynb"
                    onChange={handleJupyterUpload}
                    className="flex-1"
                  />
                  <Upload className="h-5 w-5 text-blue-500" />
                </div>
              </div>

              {/* Model Weights Input */}
              <div className="box">
                <label className="label">Model Weights</label>
                <Textarea
                  placeholder="Enter model weights configuration..."
                  value={modelWeights}
                  onChange={(e) => setModelWeights(e.target.value)}
                  className="min-h-[100px]"
                />
              </div>

              {/* Model Details Input */}
              <div className="box">
                <label className="label">Model Details</label>
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
              className="submit-btn"
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

          {/* Error Message */}
          {error && (
            <Alert variant="destructive" className="alert">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default CodeAlchemist;

