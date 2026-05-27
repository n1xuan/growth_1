package main

import (
	"math"
	"testing"

	"github.com/go-gl/mathgl/mgl64"
)

func TestParseFloatList(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    []float64
		wantErr bool
	}{
		{
			name:    "empty string",
			input:   "",
			want:    nil,
			wantErr: false,
		},
		{
			name:    "single value",
			input:   "90.0",
			want:    []float64{90.0},
			wantErr: false,
		},
		{
			name:    "multiple values",
			input:   "0,45,90,135",
			want:    []float64{0, 45, 90, 135},
			wantErr: false,
		},
		{
			name:    "values with spaces",
			input:   "0, 45, 90, 135",
			want:    []float64{0, 45, 90, 135},
			wantErr: false,
		},
		{
			name:    "decimal values",
			input:   "0.5,45.25,90.75",
			want:    []float64{0.5, 45.25, 90.75},
			wantErr: false,
		},
		{
			name:    "negative values",
			input:   "-45,0,45",
			want:    []float64{-45, 0, 45},
			wantErr: false,
		},
		{
			name:    "invalid value",
			input:   "0,abc,90",
			want:    nil,
			wantErr: true,
		},
		{
			name:    "trailing comma",
			input:   "0,45,90,",
			want:    []float64{0, 45, 90},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseFloatList(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("parseFloatList() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if len(got) != len(tt.want) {
					t.Errorf("parseFloatList() length = %v, want %v", len(got), len(tt.want))
					return
				}
				for i := range got {
					if math.Abs(got[i]-tt.want[i]) > 1e-9 {
						t.Errorf("parseFloatList()[%d] = %v, want %v", i, got[i], tt.want[i])
					}
				}
			}
		})
	}
}

func TestComputeCameraFromAngles(t *testing.T) {
	tests := []struct {
		name         string
		azimuthalDeg float64
		polarDeg     float64
		R            float64
		checkEye     func(t *testing.T, eye mgl64.Vec3)
	}{
		{
			name:         "azimuthal 0, polar 90 (equatorial plane, positive x)",
			azimuthalDeg: 0,
			polarDeg:     90,
			R:            4.0,
			checkEye: func(t *testing.T, eye mgl64.Vec3) {
				// Should be at (R, 0, 0) for azimuthal=0, polar=90
				expected := mgl64.Vec3{4.0, 0, 0}
				diff := eye.Sub(expected)
				dist := math.Sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2])
				if dist > 1e-6 {
					t.Errorf("Expected eye position ~(%v, %v, %v), got (%v, %v, %v)",
						expected[0], expected[1], expected[2], eye[0], eye[1], eye[2])
				}
				// Check distance from origin
				eyeLen := math.Sqrt(eye[0]*eye[0] + eye[1]*eye[1] + eye[2]*eye[2])
				if math.Abs(eyeLen-4.0) > 1e-6 {
					t.Errorf("Expected distance from origin = 4.0, got %v", eyeLen)
				}
			},
		},
		{
			name:         "azimuthal 90, polar 90 (equatorial plane, positive y)",
			azimuthalDeg: 90,
			polarDeg:     90,
			R:            4.0,
			checkEye: func(t *testing.T, eye mgl64.Vec3) {
				// Should be at (0, R, 0) for azimuthal=90, polar=90
				expected := mgl64.Vec3{0, 4.0, 0}
				diff := eye.Sub(expected)
				dist := math.Sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2])
				if dist > 1e-6 {
					t.Errorf("Expected eye position ~(%v, %v, %v), got (%v, %v, %v)",
						expected[0], expected[1], expected[2], eye[0], eye[1], eye[2])
				}
			},
		},
		{
			name:         "azimuthal 0, polar 0 (north pole)",
			azimuthalDeg: 0,
			polarDeg:     0,
			R:            4.0,
			checkEye: func(t *testing.T, eye mgl64.Vec3) {
				// Should be at (0, 0, R) for polar=0
				expected := mgl64.Vec3{0, 0, 4.0}
				diff := eye.Sub(expected)
				dist := math.Sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2])
				if dist > 1e-6 {
					t.Errorf("Expected eye position ~(%v, %v, %v), got (%v, %v, %v)",
						expected[0], expected[1], expected[2], eye[0], eye[1], eye[2])
				}
				// Note: At north pole, the camera matrix may be degenerate (gimbal lock)
				// This is acceptable for this use case
			},
		},
		{
			name:         "distance check",
			azimuthalDeg: 45,
			polarDeg:     60,
			R:            5.0,
			checkEye: func(t *testing.T, eye mgl64.Vec3) {
				// Distance from origin should be R
				eyeLen := math.Sqrt(eye[0]*eye[0] + eye[1]*eye[1] + eye[2]*eye[2])
				if math.Abs(eyeLen-5.0) > 1e-6 {
					t.Errorf("Expected distance from origin = 5.0, got %v", eyeLen)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			eye, camera := computeCameraFromAngles(tt.azimuthalDeg, tt.polarDeg, tt.R)
			tt.checkEye(t, eye)

			// Check that camera matrix is valid (4x4)
			// Mat4 is always 4x4, so we just check that it's not all zeros
			// Skip this check for polar=0 (north pole) due to gimbal lock
			if tt.polarDeg != 0 {
				hasNonZero := false
				for i := 0; i < 4; i++ {
					for j := 0; j < 4; j++ {
						if math.Abs(camera.At(i, j)) > 1e-10 {
							hasNonZero = true
							break
						}
					}
					if hasNonZero {
						break
					}
				}
				if !hasNonZero {
					t.Error("Camera matrix should not be all zeros")
				}
			}
		})
	}
}

func TestGenerateCameraAngles(t *testing.T) {
	tests := []struct {
		name        string
		num_images  int
		job_num     int
		jobs_modulo int
		out_of_plane bool
		polar_angle float64
		wantCount   int
		checkAngles func(t *testing.T, angles []CameraAngle)
	}{
		{
			name:        "equispaced, 4 images",
			num_images:  4,
			job_num:     0,
			jobs_modulo: 1,
			out_of_plane: false,
			polar_angle: 90.0,
			wantCount:   4,
			checkAngles: func(t *testing.T, angles []CameraAngle) {
				if len(angles) != 4 {
					t.Errorf("Expected 4 angles, got %d", len(angles))
				}
				// Check azimuthal angles are equispaced starting from 90
				expectedAzimuthals := []float64{90.0, 180.0, 270.0, 360.0}
				for i, angle := range angles {
					if math.Abs(angle.Azimuthal-expectedAzimuthals[i]) > 1e-6 {
						t.Errorf("Angle[%d].Azimuthal = %v, want %v", i, angle.Azimuthal, expectedAzimuthals[i])
					}
					if math.Abs(angle.Polar-90.0) > 1e-6 {
						t.Errorf("Angle[%d].Polar = %v, want 90.0", i, angle.Polar)
					}
				}
			},
		},
		{
			name:        "custom polar angle",
			num_images:  2,
			job_num:     0,
			jobs_modulo: 1,
			out_of_plane: false,
			polar_angle: 45.0,
			wantCount:   2,
			checkAngles: func(t *testing.T, angles []CameraAngle) {
				for i, angle := range angles {
					if math.Abs(angle.Polar-45.0) > 1e-6 {
						t.Errorf("Angle[%d].Polar = %v, want 45.0", i, angle.Polar)
					}
				}
			},
		},
		{
			name:        "jobs_modulo filtering",
			num_images:  8,
			job_num:     1,
			jobs_modulo: 2,
			out_of_plane: false,
			polar_angle: 90.0,
			wantCount:   4,
			checkAngles: func(t *testing.T, angles []CameraAngle) {
				if len(angles) != 4 {
					t.Errorf("Expected 4 angles, got %d", len(angles))
				}
				// Should start at index 1, then 3, 5, 7
				// dth = 360/8 = 45, so indices 1,3,5,7 give: 90+1*45=135, 90+3*45=225, 90+5*45=315, 90+7*45=405
				expectedAzimuthals := []float64{135.0, 225.0, 315.0, 405.0}
				for i, angle := range angles {
					if math.Abs(angle.Azimuthal-expectedAzimuthals[i]) > 1e-6 {
						t.Errorf("Angle[%d].Azimuthal = %v, want %v", i, angle.Azimuthal, expectedAzimuthals[i])
					}
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			angles := generateCameraAngles(tt.num_images, tt.job_num, tt.jobs_modulo, tt.out_of_plane, tt.polar_angle)
			if len(angles) != tt.wantCount {
				t.Errorf("generateCameraAngles() returned %d angles, want %d", len(angles), tt.wantCount)
			}
			if tt.checkAngles != nil {
				tt.checkAngles(t, angles)
			}
		})
	}
}

func TestCameraAngleStruct(t *testing.T) {
	angle := CameraAngle{
		Azimuthal: 45.0,
		Polar:     90.0,
	}

	if angle.Azimuthal != 45.0 {
		t.Errorf("Azimuthal = %v, want 45.0", angle.Azimuthal)
	}
	if angle.Polar != 90.0 {
		t.Errorf("Polar = %v, want 90.0", angle.Polar)
	}
}

