
#include "coupler.h"
#include "dynamics_rk.h"
#include "time_averager.h"
#include "sc_init.h"
#include "sc_perturb.h"
#include "les_closure.h"
#include "surface_flux.h"
#include "sponge_layer.h"
#include "YAKL_netcdf.h"


/*
In blender, delete the initial objects.
Import opensteetmap, buildings only, as separate objects.
Then rotate to align with your grid and delete what you want.
Export to obj with all options turn off except for triangulate faces turned on, Y Forward, Z Up.
We only want triangle faces for simplicity.
This code will handle the rest.
*/


struct Vertex {
  float x, y, z;
  inline friend std::ostream &operator<<(std::ostream& os, Vertex const &v ) {
    os << "[" << v.x << " , " << v.y << " , " << v.z << "]";
    return os;
  }
};

struct Face {
  Vertex v1, v2, v3;
};

struct Mesh {
  typedef yakl::Array<Face,1,yakl::memDevice,yakl::styleC> Faces;
  Faces  faces;
  Vertex domain_lo, domain_hi;
  void add_offset(float x = 0, float y = 0, float z = 0) {
    yakl::c::parallel_for( YAKL_AUTO_LABEL() , faces.size() , KOKKOS_LAMBDA (int i) {
      faces(i).v1.x += x;    faces(i).v1.y += y;    faces(i).v1.z += z;
      faces(i).v2.x += x;    faces(i).v2.y += y;    faces(i).v2.z += z;
      faces(i).v3.x += x;    faces(i).v3.y += y;    faces(i).v3.z += z;
    });
    domain_lo.x += x;    domain_lo.y += y;    domain_lo.z += z;
    domain_hi.x += x;    domain_hi.y += y;    domain_hi.z += z;
  }
  inline friend std::ostream &operator<<(std::ostream& os, Mesh const &m ) {
    std::cout << "Bounding Box:    " << m.domain_lo << " x " << m.domain_hi << "\n";
    std::cout << "Number of faces: " << m.faces.size() << std::endl;
    return os;
  }
};

inline Mesh read_obj_mesh(std::string fname) {
  float constexpr pos_huge = std::numeric_limits<float>::max();
  float constexpr neg_huge = std::numeric_limits<float>::lowest();
  float xl = pos_huge, yl = pos_huge, zl = pos_huge;
  float xh = neg_huge, yh = neg_huge, zh = neg_huge;
  std::ifstream file(fname);
  std::string line;
  std::vector<Vertex> vertices;
  std::vector<Face>   faces;
  Mesh mesh;
  while (std::getline(file, line)) {
    if (line.size() > 0) {
      if (line[0] == 'v') {
        std::string lab;
        float x, y, z;
        std::stringstream(line) >> lab >> x >> y >> z;
        xl = std::min(xl,x);  yl = std::min(yl,y);  zl = std::min(zl,z);
        xh = std::max(xh,x);  yh = std::max(yh,y);  zh = std::max(zh,z);
        vertices.push_back({x,y,z});
      }
      if (line[0] == 'f') {
        std::string lab;
        int i, j, k;
        std::stringstream(line) >> lab >> i >> j >> k;
        faces.push_back({vertices.at(i-1),vertices.at(j-1),vertices.at(k-1)});
      }
    }
  }
  mesh.domain_lo = {xl,yl,zl};
  mesh.domain_hi = {xh,yh,zh};
  auto mesh_faces_host = Mesh::Faces("faces",faces.size()).createHostObject();
  for (int i=0; i < faces.size(); i++) { mesh_faces_host(i) = faces[i]; }
  mesh.faces = mesh_faces_host.createDeviceCopy();
  file.close();
  return mesh;
}

KOKKOS_INLINE_FUNCTION float sign(Vertex const &v1, Vertex const &v2, Vertex const &v3) {
  return (v1.x-v3.x)*(v2.y-v3.y) - (v2.x-v3.x)*(v1.y-v3.y);
}

KOKKOS_INLINE_FUNCTION bool point_in_triangle(Vertex pt, Face const &face) {
  float d1 = sign( pt , face.v1 , face.v2 );
  float d2 = sign( pt , face.v2 , face.v3 );
  float d3 = sign( pt , face.v3 , face.v1 );
  bool  has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
  bool  has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);
  return !(has_neg && has_pos);
}

KOKKOS_INLINE_FUNCTION float bilinear_interpolation( Face const &face , float x , float y ) {
  auto v1 = face.v1;
  auto v2 = face.v2;
  auto v3 = face.v3;
  // Calculate the area of the triangle
  float area = 0.5f * std::abs(v1.x * (v2.y - v3.y) + v2.x * (v3.y - v1.y) + v3.x * (v1.y - v2.y));
  // Calculate the barycentric coordinates
  float w1 = (v2.x * v3.y - v3.x * v2.y + (v2.y - v3.y) * x + (v3.x - v2.x) * y) / (2 * area);
  float w2 = (v3.x * v1.y - v1.x * v3.y + (v3.y - v1.y) * x + (v1.x - v3.x) * y) / (2 * area);
  float w3 = 1 - w1 - w2;
  // Interpolate the z value
  return w1 * v1.z + w2 * v2.z + w3 * v3.z;
}



int main(int argc, char** argv) {
  MPI_Init( &argc , &argv );
  Kokkos::initialize();
  yakl::init();
  {
    yakl::timer_start("main");

    auto mesh = read_obj_mesh("/home/imn/nyc2.obj");
    mesh.add_offset( -mesh.domain_lo.x , -mesh.domain_lo.y , -mesh.domain_lo.z );
    std::cout << mesh;
    float dx = 5;
    float dy = 5;
    int nx = (int) std::ceil(mesh.domain_hi.x/dx);
    int ny = (int) std::ceil(mesh.domain_hi.y/dy);
    std::cout << nx << " , " << ny << std::endl;
    floatHost2d heightmap("heightmap",ny,nx);
    heightmap = mesh.domain_lo.z;
    yakl::timer_start("heightmap");
    yakl::c::parallel_for( YAKL_AUTO_LABEL() , yakl::c::SimpleBounds<2>(ny,nx) , KOKKOS_LAMBDA (int j, int i) {
      Vertex pt( { (i+0.5f)*dx , (j+0.5f)*dy , 0.f } );
      for (int k=0; k < mesh.faces.size(); k++) {
        if ( point_in_triangle( pt , mesh.faces(k) ) ) {
          heightmap(j,i) = std::max( heightmap(j,i) , bilinear_interpolation( mesh.faces(k) , pt.x , pt.y ) );
        }
      }
    });
    yakl::timer_stop("heightmap");

    yakl::SimpleNetCDF nc;
    nc.create( "heightmap.nc");
    nc.createDim( "x" , nx );
    nc.createDim( "y" , ny );
    nc.write( heightmap , "heightmap" , {"y","x"} );
    nc.close();

    yakl::timer_stop("main");
  }
  yakl::finalize();
  Kokkos::finalize();
  MPI_Finalize();
}

