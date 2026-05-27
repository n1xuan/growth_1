// Package: main
// File: api.go
// Description: C-compatible API for Python bindings using cgo.
//
// This file provides exported functions that can be called from Python via ctypes.
// Functions use JSON for parameter passing to simplify the interface.
//
// Author: Ivan Grega
// License: MIT

package main

/*
#include <stdlib.h>
#include <string.h>
*/
import "C"
import (
	"encoding/json"
	"os"
	"unsafe"

	"github.com/igrega348/xray_projection_render/deformations"
	"github.com/igrega348/xray_projection_render/objects"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

// RenderParams represents all parameters needed for rendering.
type RenderParams struct {
	Input             string        `json:"input"`
	OutputDir         string        `json:"output_dir"`
	FnamePattern      string        `json:"fname_pattern"`
	Resolution        int           `json:"resolution"`
	NumImages         int           `json:"num_images"`
	OutOfPlane        bool          `json:"out_of_plane"`
	DS                float64       `json:"ds"`
	R                 float64       `json:"R"`
	FOV               float64       `json:"fov"`
	JobsModulo        int           `json:"jobs_modulo"`
	JobNum            int           `json:"job_num"`
	TransformsFile    string        `json:"transforms_file"`
	DeformationFile   string        `json:"deformation_file"`
	TimeLabel         float64       `json:"time_label"`
	Transparency      bool          `json:"transparency"`
	ExportVolume      bool          `json:"export_volume"`
	PolarAngle        float64       `json:"polar_angle"`
	CameraAngles      []CameraAngle `json:"camera_angles"`
	DensityMultiplier float64       `json:"density_multiplier"`
	FlatField         float64       `json:"flat_field"`
	Integration       string        `json:"integration"`
	LogLevel          string        `json:"log_level"` // "trace", "debug", "info", "warn", "error", "fatal", "panic", or "disabled"
}

// RenderResult represents the result of a render operation.
type RenderResult struct {
	Success   bool   `json:"success"`
	Error     string `json:"error,omitempty"`
	NumImages int    `json:"num_images"`
	OutputDir string `json:"output_dir"`
}

// RenderProjections renders X-ray projections based on JSON parameters.
// Parameters:
//   - jsonParams: JSON string containing RenderParams
//
// Returns:
//   - JSON string containing RenderResult
//   - Memory is allocated using C.malloc and must be freed by the caller
//
//export RenderProjections
func RenderProjections(jsonParams *C.char) *C.char {
	paramsStr := C.GoString(jsonParams)

	var params RenderParams
	if err := json.Unmarshal([]byte(paramsStr), &params); err != nil {
		result := RenderResult{
			Success: false,
			Error:   "Failed to parse parameters: " + err.Error(),
		}
		resultJSON, _ := json.Marshal(result)
		return C.CString(string(resultJSON))
	}

	// Set logging level
	logLevel := params.LogLevel
	if logLevel == "" {
		logLevel = "error" // Default to quiet (only errors)
	}
	setLogLevel(logLevel)

	// Set global variables from params
	density_multiplier = params.DensityMultiplier
	flat_field = params.FlatField
	if params.Integration == "simple" {
		integrate = integrate_along_ray
	} else {
		integrate = integrate_hierarchical
	}

	// Reset global state
	lat = []objects.Object{}
	df = []deformations.Deformation{}
	warned_clipping_max = false
	warned_clipping_min = false

	// Call render function with provided parameters
	// Wrap in a panic recovery since render may call log.Fatal
	defer func() {
		if r := recover(); r != nil {
			// Panic was recovered, but we can't return from here
			// The result will be set below
		}
	}()

	// Note: render() may call log.Fatal which will terminate the program.
	// This is expected behavior for CLI usage. For API usage, we rely on
	// the caller to ensure parameters are valid.
	render(
		params.Input,
		params.OutputDir,
		params.FnamePattern,
		params.Resolution,
		params.NumImages,
		params.OutOfPlane,
		params.DS,
		params.R,
		params.FOV,
		params.JobsModulo,
		params.JobNum,
		params.TransformsFile,
		params.DeformationFile,
		params.TimeLabel,
		params.Transparency,
		params.ExportVolume,
		params.PolarAngle,
		params.CameraAngles,
	)

	result := RenderResult{
		Success:   true,
		NumImages: params.NumImages,
		OutputDir: params.OutputDir,
	}
	resultJSON, err := json.Marshal(result)
	if err != nil {
		errorResult := RenderResult{
			Success: false,
			Error:   "Failed to marshal result: " + err.Error(),
		}
		errorJSON, _ := json.Marshal(errorResult)
		return C.CString(string(errorJSON))
	}

	return C.CString(string(resultJSON))
}

// FreeString frees a C string allocated by RenderProjections.
// This should be called from Python after using the returned string.
//
//export FreeString
func FreeString(str *C.char) {
	C.free(unsafe.Pointer(str))
}

// setLogLevel sets the zerolog global log level based on a string.
// Valid levels: "trace", "debug", "info", "warn", "error", "fatal", "panic", "disabled"
// Defaults to "error" if an invalid level is provided.
func setLogLevel(levelStr string) {
	// Configure logger to write to stderr (not stdout) to avoid interfering with output
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})

	var level zerolog.Level
	switch levelStr {
	case "trace":
		level = zerolog.TraceLevel
	case "debug":
		level = zerolog.DebugLevel
	case "info":
		level = zerolog.InfoLevel
	case "warn":
		level = zerolog.WarnLevel
	case "error":
		level = zerolog.ErrorLevel
	case "fatal":
		level = zerolog.FatalLevel
	case "panic":
		level = zerolog.PanicLevel
	case "disabled":
		level = zerolog.Disabled
	default:
		level = zerolog.ErrorLevel // Default to quiet
	}
	zerolog.SetGlobalLevel(level)
}
