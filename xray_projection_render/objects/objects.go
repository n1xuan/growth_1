package objects

import (
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/go-gl/mathgl/mgl64"
	"github.com/rs/zerolog/log"
)

type Object interface {
	Density(x, y, z float64) float64
	ToMap() map[string]interface{}
	FromMap(data map[string]interface{}) error
	MinFeatureSize() float64
	String() string
}

type Sphere struct {
	Object
	// parameters are center and radius
	Center mgl64.Vec3
	Radius float64
	Rho    float64
}

func (s *Sphere) String() string {
	return fmt.Sprintf("Sphere{Center: %v, Radius: %v, Rho: %v}", s.Center, s.Radius, s.Rho)
}

func (s *Sphere) ToMap() map[string]interface{} {
	return map[string]interface{}{
		"type":   "sphere",
		"center": s.Center,
		"radius": s.Radius,
		"rho":    s.Rho,
	}
}

func (s *Sphere) FromMap(data map[string]interface{}) error {
	var ok bool
	var err error
	var slice []interface{}
	if slice, ok = data["center"].([]interface{}); !ok {
		return fmt.Errorf("center is not a Vec3")
	}
	for i, val := range slice {
		if s.Center[i], err = ToFloat64(val); err != nil {
			return fmt.Errorf("center[%d] is not a float64", i)
		}
	}
	if s.Radius, ok = data["radius"].(float64); !ok {
		return fmt.Errorf("radius is not a float64")
	}
	if s.Rho, ok = data["rho"].(float64); !ok {
		return fmt.Errorf("rho is not a float64")
	}
	return nil
}

func (s *Sphere) Density(x, y, z float64) float64 {
	x = x - s.Center[0]
	y = y - s.Center[1]
	z = z - s.Center[2]
	r_2 := x*x + y*y + z*z
	if r_2 < s.Radius*s.Radius {
		return s.Rho
	}
	return 0.0
}

func (s *Sphere) MinFeatureSize() float64 {
	return s.Radius
}

type Cube struct {
	Object
	// parameters are center and side length
	Center mgl64.Vec3
	Side   float64
	Rho    float64
	Box    Box
}

func (c *Cube) String() string {
	return fmt.Sprintf("Cube{Center: %v, Side: %v, Rho: %v}", c.Center, c.Side, c.Rho)
}

func (c *Cube) ToMap() map[string]interface{} {
	return map[string]interface{}{
		"type":   "cube",
		"center": c.Center,
		"side":   c.Side,
		"rho":    c.Rho,
	}
}

func (c *Cube) FromMap(data map[string]interface{}) error {
	var ok bool
	var slice []interface{}
	if slice, ok = data["center"].([]interface{}); !ok {
		return fmt.Errorf("center is not a Vec3")
	}
	for i, val := range slice {
		c.Center[i] = val.(float64)
	}
	if c.Side, ok = data["side"].(float64); !ok {
		return fmt.Errorf("side is not a float64")
	}
	if c.Rho, ok = data["rho"].(float64); !ok {
		return fmt.Errorf("rho is not a float64")
	}
	c.Box = Box{Center: c.Center, Sides: mgl64.Vec3{c.Side, c.Side, c.Side}, Rho: c.Rho}
	return nil
}

func (c *Cube) Density(x, y, z float64) float64 {
	return c.Box.Density(x, y, z)
}

func (c *Cube) MinFeatureSize() float64 {
	return c.Box.MinFeatureSize()
}

type Box struct {
	Object
	// parameters are center and side lengths
	Center mgl64.Vec3
	Sides  mgl64.Vec3
	Rho    float64
}

func (b *Box) String() string {
	return fmt.Sprintf("Box{Center: %v, Sides: %v, Rho: %v}", b.Center, b.Sides, b.Rho)
}

func (b *Box) ToMap() map[string]interface{} {
	return map[string]interface{}{
		"type":   "box",
		"center": b.Center,
		"sides":  b.Sides,
		"rho":    b.Rho,
	}
}

func (b *Box) FromMap(data map[string]interface{}) error {
	var ok bool
	var slice []interface{}
	if slice, ok = data["center"].([]interface{}); !ok {
		return fmt.Errorf("center is not a Vec3")
	}
	err := ToVec(&slice, &b.Center)
	if err != nil {
		return err
	}
	if slice, ok = data["sides"].([]interface{}); !ok {
		return fmt.Errorf("sides is not a Vec3")
	}
	err = ToVec(&slice, &b.Sides)
	if err != nil {
		return err
	}
	if b.Rho, err = ToFloat64(data["rho"]); err != nil {
		return fmt.Errorf("rho is not a float64")
	}
	return nil
}

func (b *Box) Density(x, y, z float64) float64 {
	x = math.Abs(x - b.Center[0])
	y = math.Abs(y - b.Center[1])
	z = math.Abs(z - b.Center[2])
	if x < 0.5*b.Sides[0] && y < 0.5*b.Sides[1] && z < 0.5*b.Sides[2] {
		return b.Rho
	}
	return 0.0
}

func (b *Box) MinFeatureSize() float64 {
	return 0.1 * math.Min(b.Sides[0], math.Min(b.Sides[1], b.Sides[2]))
}

type Parallelepiped struct {
	Object
	// parameters are origin and vectors for sides
	Origin     mgl64.Vec3
	V0, V1, V2 mgl64.Vec3
	Rho        float64
	mat        mgl64.Mat3 // matrix for coordinate transformation
}

func (p *Parallelepiped) String() string {
	return fmt.Sprintf("Parallelepiped{Origin: %v, V0: %v, V1: %v, V2: %v, Rho: %v}", p.Origin, p.V0, p.V1, p.V2, p.Rho)
}

func (p *Parallelepiped) ToMap() map[string]interface{} {
	return map[string]interface{}{
		"type":   "parallelepiped",
		"origin": p.Origin,
		"v0":     p.V0,
		"v1":     p.V1,
		"v2":     p.V2,
		"rho":    p.Rho,
	}
}

func (p *Parallelepiped) FromMap(data map[string]interface{}) error {
	var ok bool
	var slice []interface{}
	if slice, ok = data["origin"].([]interface{}); !ok {
		return fmt.Errorf("origin is not a Vec3")
	}
	err := ToVec(&slice, &p.Origin)
	if err != nil {
		return err
	}
	if slice, ok = data["v0"].([]interface{}); !ok {
		return fmt.Errorf("v0 is not a Vec3")
	}
	err = ToVec(&slice, &p.V0)
	if err != nil {
		return err
	}
	if slice, ok = data["v1"].([]interface{}); !ok {
		return fmt.Errorf("v1 is not a Vec3")
	}
	err = ToVec(&slice, &p.V1)
	if err != nil {
		return err
	}
	if slice, ok = data["v2"].([]interface{}); !ok {
		return fmt.Errorf("v2 is not a Vec3")
	}
	err = ToVec(&slice, &p.V2)
	if err != nil {
		return err
	}
	if p.Rho, err = ToFloat64(data["rho"]); err != nil {
		return fmt.Errorf("rho is not a float64")
	}
	p.mat = mgl64.Mat3FromCols(p.V0, p.V1, p.V2).Inv()
	return nil
}

func (p *Parallelepiped) Density(x, y, z float64) float64 {
	// transform point to parallelepiped coordinates
	pt := mgl64.Vec3{x, y, z}
	x, y, z = p.mat.Mul3x1(pt.Sub(p.Origin)).Elem()
	if x > 0.0 && x < 1.0 && y > 0.0 && y < 1.0 && z > 0.0 && z < 1.0 {
		return p.Rho
	}
	return 0.0
}

func (p *Parallelepiped) MinFeatureSize() float64 {
	return 0.2 * math.Min(p.V0.Len(), math.Min(p.V1.Len(), p.V2.Len()))
}

func ToFloat64(data interface{}) (float64, error) {
	switch t := data.(type) {
	case int:
		return float64(t), nil
	case float64:
		return t, nil
	default:
		return 0.0, fmt.Errorf("data is not a float64")
	}
}

func ToVec(data *[]interface{}, vec *mgl64.Vec3) error {
	for i, val := range *data {
		switch t := val.(type) {
		case int:
			vec[i] = float64(t)
		case float64:
			vec[i] = t
		}
	}
	return nil
}

type Cylinder struct {
	Object
	// cylinder is a line segment with thickness
	P0, P1 mgl64.Vec3
	Radius float64
	Rho    float64
}

func (c *Cylinder) String() string {
	return fmt.Sprintf("Cylinder{P0: %v, P1: %v, Radius: %v, Rho: %v}", c.P0, c.P1, c.Radius, c.Rho)
}

func (c *Cylinder) ToMap() map[string]interface{} {
	return map[string]interface{}{
		"type":   "cylinder",
		"p0":     c.P0,
		"p1":     c.P1,
		"radius": c.Radius,
		"rho":    c.Rho,
	}
}

func (c *Cylinder) FromMap(data map[string]interface{}) error {
	var ok bool
	var slice []interface{}
	if slice, ok = data["p0"].([]interface{}); !ok {
		return fmt.Errorf("p0 is not a Vec3")
	}
	err := ToVec(&slice, &c.P0)
	if err != nil {
		return err
	}
	if slice, ok = data["p1"].([]interface{}); !ok {
		return fmt.Errorf("p0 is not a Vec3")
	}
	err = ToVec(&slice, &c.P1)
	if err != nil {
		return err
	}
	if c.Radius, ok = data["radius"].(float64); !ok {
		return fmt.Errorf("radius is not a float64")
	}
	if _, ok := data["rho"]; !ok {
		c.Rho = 1.0
	} else if c.Rho, err = ToFloat64(data["rho"]); err != nil {
		return fmt.Errorf("rho is not a float64")
	}
	return nil
}

func (cyl *Cylinder) Density(x, y, z float64) float64 {
	// get the vector from the point to the line
	v := cyl.P1.Sub(cyl.P0)
	w := mgl64.Vec3{x, y, z}.Sub(cyl.P0)
	// get the projection of w onto v
	c := w.Dot(v) / v.Dot(v)
	if c < 0.0 || c > 1.0 { // point is definitely not on the line
		return 0.0
	}
	// get the distance from the point to the line
	d := w.Sub(v.Mul(c)).Len()
	if d < cyl.Radius {
		return cyl.Rho
	} else {
		return 0.0
	}
}

func (cyl *Cylinder) MinFeatureSize() float64 {
	return cyl.Radius
}

type ObjectCollection struct {
	Object
	Objects        []Object
	GreedyDensEval bool
}

func (oc *ObjectCollection) String() string {
	if len(oc.Objects) > 5 {
		return fmt.Sprintf("ObjectCollection with %d objects. GreedyDensEval=%v", len(oc.Objects), oc.GreedyDensEval)
	} else {
		return fmt.Sprintf("ObjectCollection{%v, GreedyDensEval=%v}", oc.Objects, oc.GreedyDensEval)
	}
}

func (oc *ObjectCollection) ToMap() map[string]interface{} {
	var objects = make([]map[string]interface{}, len(oc.Objects))
	for i, object := range oc.Objects {
		objects[i] = object.ToMap()
	}
	return map[string]interface{}{
		"type":    "object_collection",
		"objects": objects,
	}
}

func (oc *ObjectCollection) FromMap(data map[string]interface{}) error {
	var objects []Object
	if greedy_dens_eval, ok := data["greedy_dens_eval"].(bool); ok {
		log.Info().Msgf("Setting greedy dens eval to %v", greedy_dens_eval)
		oc.GreedyDensEval = greedy_dens_eval
	}
	if objects_data, ok := data["objects"].([]interface{}); ok {
		objects = make([]Object, len(objects_data))
		log.Info().Msgf("Loading object collection with %d objects", len(objects_data))
		for i, object_data := range objects_data {
			switch object_data.(map[string]interface{})["type"] {
			case "sphere":
				objects[i] = &Sphere{}
			case "cube":
				objects[i] = &Cube{}
			case "box":
				objects[i] = &Box{}
			case "cylinder":
				objects[i] = &Cylinder{}
			case "parallelepiped":
				objects[i] = &Parallelepiped{}
			case "gyroid":
				objects[i] = &Gyroid{}
			case "tessellated_obj_coll":
				objects[i] = &TessellatedObjColl{}
			case "voxel_grid":
				objects[i] = &VoxelGrid{}
			default:
				return fmt.Errorf("unknown object type")
			}
			if err := objects[i].FromMap(object_data.(map[string]interface{})); err != nil {
				return err
			}
		}
	} else {
		return fmt.Errorf("objects is not a list")
	}
	oc.Objects = objects
	return nil
}

func (oc *ObjectCollection) Density(x, y, z float64) float64 {
	var density float64
	for _, object := range oc.Objects {
		rho := object.Density(x, y, z)
		if oc.GreedyDensEval && rho > 0.0 {
			return rho
		}
		density += rho
	}
	// clip between 0 and 1
	if density < 0.0 {
		density = 0.0
	} else if density > 1.0 {
		density = 1.0
	}
	return density
}

func (oc *ObjectCollection) MinFeatureSize() float64 {
	out := math.Inf(1)
	for _, object := range oc.Objects {
		out = math.Min(out, object.MinFeatureSize())
	}
	return out
}

type UnitCell struct {
	// object collection. But overload density method and provide bounds
	Objects                            ObjectCollection
	Xmin, Xmax, Ymin, Ymax, Zmin, Zmax float64
}

func (uc *UnitCell) String() string {
	return fmt.Sprintf("UnitCell{Objects: {%v}, Xmin: %v, Xmax: %v, Ymin: %v, Ymax: %v, Zmin: %v, Zmax: %v}", uc.Objects.String(), uc.Xmin, uc.Xmax, uc.Ymin, uc.Ymax, uc.Zmin, uc.Zmax)
}

func (uc *UnitCell) Density(x, y, z float64) float64 {
	// check if point is within bounds. But account for objects a bit smaller
	if x < uc.Xmin || x > uc.Xmax || y < uc.Ymin || y > uc.Ymax || z < uc.Zmin || z > uc.Zmax {
		return 0.0
	}
	return uc.Objects.Density(x, y, z)
}

func (uc *UnitCell) ToMap() map[string]interface{} {
	return map[string]interface{}{
		"type":    "unit_cell",
		"objects": uc.Objects.ToMap(),
		"xmin":    uc.Xmin,
		"xmax":    uc.Xmax,
		"ymin":    uc.Ymin,
		"ymax":    uc.Ymax,
		"zmin":    uc.Zmin,
		"zmax":    uc.Zmax,
	}
}

func (uc *UnitCell) FromMap(data map[string]interface{}) error {
	var err error
	if objects_data, ok := data["objects"].(map[string]interface{}); ok {
		objects := ObjectCollection{}
		if err := objects.FromMap(objects_data); err != nil {
			return err
		}
		uc.Objects = objects
		uc.Objects.GreedyDensEval = true
	} else {
		return fmt.Errorf("objects is not a map")
	}
	if uc.Xmin, err = ToFloat64(data["xmin"]); err != nil {
		return fmt.Errorf("xmin is not a float64")
	}
	if uc.Xmax, err = ToFloat64(data["xmax"]); err != nil {
		return fmt.Errorf("xmax is not a float64")
	}
	if uc.Ymin, err = ToFloat64(data["ymin"]); err != nil {
		return fmt.Errorf("ymin is not a float64")
	}
	if uc.Ymax, err = ToFloat64(data["ymax"]); err != nil {
		return fmt.Errorf("ymax is not a float64")
	}
	if uc.Zmin, err = ToFloat64(data["zmin"]); err != nil {
		return fmt.Errorf("zmin is not a float64")
	}
	if uc.Zmax, err = ToFloat64(data["zmax"]); err != nil {
		return fmt.Errorf("zmax is not a float64")
	}
	return nil
}

type TessellatedObjColl struct {
	Object
	// lattice is given by unit cell and bounds for tessellation
	UC                                 UnitCell
	Xmin, Xmax, Ymin, Ymax, Zmin, Zmax float64
}

func (l *TessellatedObjColl) String() string {
	return fmt.Sprintf("TessellatedObjColl{UC: {%v}, Xmin: %v, Xmax: %v, Ymin: %v, Ymax: %v, Zmin: %v, Zmax: %v}", l.UC.String(), l.Xmin, l.Xmax, l.Ymin, l.Ymax, l.Zmin, l.Zmax)
}

func (l *TessellatedObjColl) ToMap() map[string]interface{} {
	return map[string]interface{}{
		"type": "tessellated_obj_coll",
		"uc":   l.UC.ToMap(),
		"xmin": l.Xmin,
		"xmax": l.Xmax,
		"ymin": l.Ymin,
		"ymax": l.Ymax,
		"zmin": l.Zmin,
		"zmax": l.Zmax,
	}
}

func (l *TessellatedObjColl) FromMap(data map[string]interface{}) error {
	var err error
	if uc_data, ok := data["uc"].(map[string]interface{}); ok {
		uc := UnitCell{}
		if err := uc.FromMap(uc_data); err != nil {
			return err
		}
		l.UC = uc
	} else {
		return fmt.Errorf("uc is not a map")
	}
	if l.Xmin, err = ToFloat64(data["xmin"]); err != nil {
		return fmt.Errorf("xmin is not a float64")
	}
	if l.Xmax, err = ToFloat64(data["xmax"]); err != nil {
		return fmt.Errorf("xmax is not a float64")
	}
	if l.Ymin, err = ToFloat64(data["ymin"]); err != nil {
		return fmt.Errorf("ymin is not a float64")
	}
	if l.Ymax, err = ToFloat64(data["ymax"]); err != nil {
		return fmt.Errorf("ymax is not a float64")
	}
	if l.Zmin, err = ToFloat64(data["zmin"]); err != nil {
		return fmt.Errorf("zmin is not a float64")
	}
	if l.Zmax, err = ToFloat64(data["zmax"]); err != nil {
		return fmt.Errorf("zmax is not a float64")
	}
	return nil
}

func (l *TessellatedObjColl) Density(x, y, z float64) float64 {
	// check if point is within bounds
	if x < l.Xmin || x > l.Xmax || y < l.Ymin || y > l.Ymax || z < l.Zmin || z > l.Zmax {
		return 0.0
	} else {
		// map point to unit cell
		dx := l.UC.Xmax - l.UC.Xmin
		x = x - dx*math.Floor((x-l.UC.Xmin)/dx)
		dy := l.UC.Ymax - l.UC.Ymin
		y = y - dy*math.Floor((y-l.UC.Ymin)/dy)
		dz := l.UC.Zmax - l.UC.Zmin
		z = z - dz*math.Floor((z-l.UC.Zmin)/dz)
		return l.UC.Density(x, y, z)
	}
}

func (l *TessellatedObjColl) MinFeatureSize() float64 {
	return l.UC.Objects.MinFeatureSize()
}

func MakeKelvin(rad float64, scale float64) UnitCell {
	var struts = []Cylinder{
		{P0: mgl64.Vec3{0.25, 0.00, 0.50}, P1: mgl64.Vec3{0.50, 0.00, 0.75}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{0.25, 1.00, 0.50}, P1: mgl64.Vec3{0.50, 1.00, 0.75}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{0.25, 0.00, 0.50}, P1: mgl64.Vec3{0.50, 0.00, 0.25}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{0.25, 1.00, 0.50}, P1: mgl64.Vec3{0.50, 1.00, 0.25}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{0.25, 0.00, 0.50}, P1: mgl64.Vec3{0.00, 0.25, 0.50}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{0.50, 0.00, 0.75}, P1: mgl64.Vec3{0.75, 0.00, 0.50}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{0.50, 1.00, 0.75}, P1: mgl64.Vec3{0.75, 1.00, 0.50}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{0.50, 0.00, 0.75}, P1: mgl64.Vec3{0.50, 0.25, 1.00}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{0.75, 0.00, 0.50}, P1: mgl64.Vec3{0.50, 0.00, 0.25}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{0.75, 1.00, 0.50}, P1: mgl64.Vec3{0.50, 1.00, 0.25}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{0.75, 0.00, 0.50}, P1: mgl64.Vec3{1.00, 0.25, 0.50}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{0.50, 0.00, 0.25}, P1: mgl64.Vec3{0.50, 0.25, 0.00}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{1.00, 0.50, 0.75}, P1: mgl64.Vec3{0.75, 0.50, 1.00}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{1.00, 0.75, 0.50}, P1: mgl64.Vec3{0.75, 1.00, 0.50}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{1.00, 0.50, 0.25}, P1: mgl64.Vec3{0.75, 0.50, 0.00}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{0.25, 1.00, 0.50}, P1: mgl64.Vec3{0.00, 0.75, 0.50}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{0.50, 1.00, 0.75}, P1: mgl64.Vec3{0.50, 0.75, 1.00}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{0.50, 1.00, 0.25}, P1: mgl64.Vec3{0.50, 0.75, 0.00}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{0.00, 0.25, 0.50}, P1: mgl64.Vec3{0.00, 0.50, 0.75}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{1.00, 0.25, 0.50}, P1: mgl64.Vec3{1.00, 0.50, 0.75}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{0.00, 0.25, 0.50}, P1: mgl64.Vec3{0.00, 0.50, 0.25}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{1.00, 0.25, 0.50}, P1: mgl64.Vec3{1.00, 0.50, 0.25}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{0.00, 0.50, 0.75}, P1: mgl64.Vec3{0.25, 0.50, 1.00}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{0.00, 0.50, 0.75}, P1: mgl64.Vec3{0.00, 0.75, 0.50}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{1.00, 0.50, 0.75}, P1: mgl64.Vec3{1.00, 0.75, 0.50}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{0.00, 0.75, 0.50}, P1: mgl64.Vec3{0.00, 0.50, 0.25}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{1.00, 0.75, 0.50}, P1: mgl64.Vec3{1.00, 0.50, 0.25}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{0.00, 0.50, 0.25}, P1: mgl64.Vec3{0.25, 0.50, 0.00}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{0.25, 0.50, 0.00}, P1: mgl64.Vec3{0.50, 0.75, 0.00}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{0.25, 0.50, 1.00}, P1: mgl64.Vec3{0.50, 0.75, 1.00}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{0.25, 0.50, 0.00}, P1: mgl64.Vec3{0.50, 0.25, 0.00}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{0.25, 0.50, 1.00}, P1: mgl64.Vec3{0.50, 0.25, 1.00}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{0.50, 0.75, 0.00}, P1: mgl64.Vec3{0.75, 0.50, 0.00}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{0.50, 0.75, 1.00}, P1: mgl64.Vec3{0.75, 0.50, 1.00}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{0.75, 0.50, 0.00}, P1: mgl64.Vec3{0.50, 0.25, 0.00}, Radius: rad, Rho: 1.0},
		{P0: mgl64.Vec3{0.75, 0.50, 1.00}, P1: mgl64.Vec3{0.50, 0.25, 1.00}, Radius: rad, Rho: 1.0},
	}
	for i := 0; i < len(struts); i++ {
		struts[i].P0 = struts[i].P0.Mul(scale)
		struts[i].P1 = struts[i].P1.Mul(scale)
	}
	var objects = make([]Object, len(struts))
	for i, strut := range struts {
		objects[i] = &strut
	}
	uc := UnitCell{Objects: ObjectCollection{Objects: objects, GreedyDensEval: true}, Xmin: 0.0, Xmax: 1.0 * scale, Ymin: 0.0, Ymax: 1.0 * scale, Zmin: 0.0, Zmax: 1.0 * scale}
	return uc
}

// func MakeOctet(rad float64) Lattice {
// 	s2 := math.Sqrt(2)
// 	var struts = []Cylinder{
// 		{P0: mgl64.Vec3{0, 0, 0}, P1: mgl64.Vec3{0.5, 0.5, -1 / s2}, Radius: rad},
// 		{P0: mgl64.Vec3{0, 0, 0}, P1: mgl64.Vec3{1, 0, 0}, Radius: rad},
// 		{P0: mgl64.Vec3{0, 0, 0}, P1: mgl64.Vec3{0.5, -0.5, -1 / s2}, Radius: rad},
// 		{P0: mgl64.Vec3{0, 0, 0}, P1: mgl64.Vec3{0, 1, 0}, Radius: rad},
// 		{P0: mgl64.Vec3{0, 0, 0}, P1: mgl64.Vec3{-0.5, 0.5, -1 / s2}, Radius: rad},
// 		{P0: mgl64.Vec3{0, 0, 0}, P1: mgl64.Vec3{0.5, 0.5, 1 / s2}, Radius: rad},
// 	}
// 	return Lattice{Objects: struts}
// }

type ObjectFactory struct{}

func (of *ObjectFactory) Create(data map[string]interface{}) (Object, error) {
	return NewObject(data)
}

func NewObject(data map[string]interface{}) (Object, error) {
	var object Object
	var err error

	// Handle regular object types
	switch data["type"] {
	case "sphere":
		object = &Sphere{}
	case "cube":
		object = &Cube{}
	case "box":
		object = &Box{}
	case "cylinder":
		object = &Cylinder{}
	case "parallelepiped":
		object = &Parallelepiped{}
	case "gyroid":
		object = &Gyroid{}
	case "object_collection":
		object = &ObjectCollection{}
	case "tessellated_obj_coll":
		object = &TessellatedObjColl{}
	case "voxel_grid":
		object = &VoxelGrid{}
	default:
		return nil, fmt.Errorf("unknown object type `%v`", data["type"])
	}
	if err = object.FromMap(data); err != nil {
		return nil, err
	}
	return object, nil
}

type VoxelGrid struct {
	Object
	Rho  []float64
	NX   int
	NY   int
	NZ   int
	Path string // Path to the original raw file
}

func (v *VoxelGrid) String() string {
	return fmt.Sprintf("VoxelGrid{NX: %d, NY: %d, NZ: %d}", v.NX, v.NY, v.NZ)
}

func (v *VoxelGrid) ToMap() map[string]interface{} {
	return map[string]interface{}{
		"type":  "voxel_grid",
		"nx":    v.NX,
		"ny":    v.NY,
		"nz":    v.NZ,
		"dtype": "float64", // Since we store as float64 internally
		"path":  v.Path,    // Path to the original raw file
	}
}

func (v *VoxelGrid) FromMap(data map[string]interface{}) error {
	var ok bool
	var err error

	// Check if this is a file path
	if path, ok := data["path"].(string); ok {
		// Check file extension
		ext := strings.ToLower(path[strings.LastIndex(path, ".")+1:])
		if ext != "raw" {
			return fmt.Errorf("only raw files are supported")
		}

		// For raw files, we need resolution information
		res_data, ok := data["resolution"].([]interface{})
		if !ok {
			return fmt.Errorf("resolution must be provided for raw files")
		}
		if len(res_data) != 3 {
			return fmt.Errorf("resolution must be a list of 3 integers")
		}
		resolution := [3]int{}
		for i, val := range res_data {
			if resolution[i], ok = val.(int); !ok {
				return fmt.Errorf("resolution[%d] is not an integer", i)
			}
		}

		// Get data type from config, default to uint8
		dtype := "uint8"
		if dtype_str, ok := data["dtype"].(string); ok {
			dtype = dtype_str
		}

		vg, err := VoxelGridFromRaw(path, resolution, dtype)
		if err != nil {
			return err
		}
		*v = *vg
		return nil
	}

	// Handle regular voxel grid creation
	if v.NX, ok = data["nx"].(int); !ok {
		return fmt.Errorf("nx is not an int")
	}
	if v.NY, ok = data["ny"].(int); !ok {
		return fmt.Errorf("ny is not an int")
	}
	if v.NZ, ok = data["nz"].(int); !ok {
		return fmt.Errorf("nz is not an int")
	}
	if v.Path, ok = data["path"].(string); !ok {
		return fmt.Errorf("path is not a string")
	}
	if rho_data, ok := data["rho"].([]interface{}); ok {
		v.Rho = make([]float64, len(rho_data))
		for i, val := range rho_data {
			if v.Rho[i], err = ToFloat64(val); err != nil {
				return fmt.Errorf("rho[%d] is not a float64", i)
			}
		}
	} else {
		return fmt.Errorf("rho is not a list")
	}
	return nil
}

func (v *VoxelGrid) Density(x, y, z float64) float64 {
	// If outside of bounds, return 0
	if x < -1 || x > 1 || y < -1 || y > 1 || z < -1 || z > 1 {
		return 0.0
	}
	// Map from [-1,1] to [0,1]
	x = (x + 1) / 2
	y = (y + 1) / 2
	z = (z + 1) / 2

	// Map to voxel coordinates
	x = x * float64(v.NX-1)
	y = y * float64(v.NY-1)
	z = z * float64(v.NZ-1)

	// Get integer coordinates
	x0 := int(math.Floor(x))
	y0 := int(math.Floor(y))
	z0 := int(math.Floor(z))
	x1 := x0 + 1
	y1 := y0 + 1
	z1 := z0 + 1

	// Clamp to bounds
	if x0 < 0 {
		x0 = 0
	}
	if y0 < 0 {
		y0 = 0
	}
	if z0 < 0 {
		z0 = 0
	}
	if x1 >= v.NX {
		x1 = v.NX - 1
	}
	if y1 >= v.NY {
		y1 = v.NY - 1
	}
	if z1 >= v.NZ {
		z1 = v.NZ - 1
	}

	// Get interpolation weights
	wx := x - float64(x0)
	wy := y - float64(y0)
	wz := z - float64(z0)

	// Get voxel values
	v000 := v.Rho[z0*v.NX*v.NY+y0*v.NX+x0]
	v001 := v.Rho[z1*v.NX*v.NY+y0*v.NX+x0]
	v010 := v.Rho[z0*v.NX*v.NY+y1*v.NX+x0]
	v011 := v.Rho[z1*v.NX*v.NY+y1*v.NX+x0]
	v100 := v.Rho[z0*v.NX*v.NY+y0*v.NX+x1]
	v101 := v.Rho[z1*v.NX*v.NY+y0*v.NX+x1]
	v110 := v.Rho[z0*v.NX*v.NY+y1*v.NX+x1]
	v111 := v.Rho[z1*v.NX*v.NY+y1*v.NX+x1]

	// Trilinear interpolation
	v00 := v000*(1-wz) + v001*wz
	v01 := v010*(1-wz) + v011*wz
	v10 := v100*(1-wz) + v101*wz
	v11 := v110*(1-wz) + v111*wz
	v0 := v00*(1-wy) + v01*wy
	v1 := v10*(1-wy) + v11*wy
	return v0*(1-wx) + v1*wx
}

func (v *VoxelGrid) MinFeatureSize() float64 {
	// Return the size of one voxel in normalized coordinates
	return 2.0 / float64(max(v.NX, max(v.NY, v.NZ)))
}

func (v *VoxelGrid) ExportToRaw(path string, res int) error {
	// Create volume grid
	volume64 := make([]float64, res*res*res)
	for i := 0; i < res; i++ {
		for j := 0; j < res; j++ {
			for k := 0; k < res; k++ {
				x := float64(i)/float64(res)*2.0 - 1.0
				y := float64(j)/float64(res)*2.0 - 1.0
				z := float64(k)/float64(res)*2.0 - 1.0
				volume64[k*res*res+i*res+j] = v.Density(x, y, z)
			}
		}
	}

	// Normalize volume to [0, 255]
	max_val := 0.0
	for i := range volume64 {
		if volume64[i] > max_val {
			max_val = volume64[i]
		}
	}
	volume := make([]byte, len(volume64))
	for i, v := range volume64 {
		volume[i] = byte(v / max_val * 255)
	}

	// Write to file
	return os.WriteFile(path, volume, 0644)
}

func VoxelGridFromRaw(path string, resolution [3]int, dtype string) (*VoxelGrid, error) {
	// Read the file
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("error reading file: %v", err)
	}

	// Calculate expected size based on data type
	var bytesPerElement int
	switch dtype {
	case "uint8":
		bytesPerElement = 1
	case "uint16":
		bytesPerElement = 2
	case "uint32":
		bytesPerElement = 4
	case "float32":
		bytesPerElement = 4
	case "float64":
		bytesPerElement = 8
	default:
		return nil, fmt.Errorf("unsupported data type: %s", dtype)
	}

	expectedSize := resolution[0] * resolution[1] * resolution[2] * bytesPerElement
	if len(data) != expectedSize {
		return nil, fmt.Errorf("file size (%d) does not match expected size (%d) for type %s", len(data), expectedSize, dtype)
	}

	// Convert bytes to float64 based on data type
	rho := make([]float64, resolution[0]*resolution[1]*resolution[2])
	switch dtype {
	case "uint8":
		for i, b := range data {
			rho[i] = float64(b) / 255.0
		}
	case "uint16":
		for i := 0; i < len(data); i += 2 {
			val := uint16(data[i]) | uint16(data[i+1])<<8
			rho[i/2] = float64(val) / 65535.0
		}
	case "uint32":
		for i := 0; i < len(data); i += 4 {
			val := uint32(data[i]) | uint32(data[i+1])<<8 | uint32(data[i+2])<<16 | uint32(data[i+3])<<24
			rho[i/4] = float64(val) / 4294967295.0
		}
	case "float32":
		for i := 0; i < len(data); i += 4 {
			bits := uint32(data[i]) | uint32(data[i+1])<<8 | uint32(data[i+2])<<16 | uint32(data[i+3])<<24
			rho[i/4] = float64(math.Float32frombits(bits))
		}
	case "float64":
		for i := 0; i < len(data); i += 8 {
			bits := uint64(data[i]) | uint64(data[i+1])<<8 | uint64(data[i+2])<<16 | uint64(data[i+3])<<24 |
				uint64(data[i+4])<<32 | uint64(data[i+5])<<40 | uint64(data[i+6])<<48 | uint64(data[i+7])<<56
			rho[i/8] = math.Float64frombits(bits)
		}
	}

	return &VoxelGrid{
		Rho:  rho,
		NX:   resolution[0],
		NY:   resolution[1],
		NZ:   resolution[2],
		Path: path,
	}, nil
}

type Gyroid struct {
	Object
	// parameters are center, scale, and thickness
	Center    mgl64.Vec3
	Scale     float64
	Thickness float64
	Rho       float64
}

func (g *Gyroid) String() string {
	return fmt.Sprintf("Gyroid{Center: %v, Scale: %v, Thickness: %v, Rho: %v}", g.Center, g.Scale, g.Thickness, g.Rho)
}

func (g *Gyroid) ToMap() map[string]interface{} {
	return map[string]interface{}{
		"type":      "gyroid",
		"center":    g.Center,
		"scale":     g.Scale,
		"thickness": g.Thickness,
		"rho":       g.Rho,
	}
}

func (g *Gyroid) FromMap(data map[string]interface{}) error {
	var ok bool
	var err error

	// Handle center - try Vec3, []interface{}, and []float64
	if vec, ok := data["center"].(mgl64.Vec3); ok {
		g.Center = vec
	} else if slice, ok := data["center"].([]interface{}); ok {
		for i, val := range slice {
			if g.Center[i], err = ToFloat64(val); err != nil {
				return fmt.Errorf("center[%d] is not a float64", i)
			}
		}
	} else if slice, ok := data["center"].([]float64); ok {
		copy(g.Center[:], slice)
	} else {
		return fmt.Errorf("center is not a Vec3")
	}

	if g.Scale, ok = data["scale"].(float64); !ok {
		return fmt.Errorf("scale is not a float64")
	}
	if g.Thickness, ok = data["thickness"].(float64); !ok {
		return fmt.Errorf("thickness is not a float64")
	}
	if g.Rho, ok = data["rho"].(float64); !ok {
		return fmt.Errorf("rho is not a float64")
	}
	return nil
}

func (g *Gyroid) Density(x, y, z float64) float64 {
	// Transform to gyroid coordinates
	x = (x - g.Center[0]) / g.Scale
	y = (y - g.Center[1]) / g.Scale
	z = (z - g.Center[2]) / g.Scale

	// Gyroid surface equation: sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) = 0
	gyroid_value := math.Sin(x)*math.Cos(y) + math.Sin(y)*math.Cos(z) + math.Sin(z)*math.Cos(x)

	// Convert to density based on thickness
	// The gyroid centre surface is where gyroid_value = 0
	if math.Abs(gyroid_value) < g.Thickness {
		// Inside the surface
		return g.Rho
	} else {
		// Outside the gyroid surface
		return 0.0
	}
}

func (g *Gyroid) MinFeatureSize() float64 {
	// The minimum feature size is related to the scale and thickness
	return g.Scale * g.Thickness * 0.1
}
