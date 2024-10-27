
#pragma once

#include "main_header.h"

namespace core {

  // Aggregate multiple fields into a single field that makes it easier to operate
  // on them together inside the same kernel. used mostly for tracers
  template <int MAX_FIELDS, class T>
  class MultipleFields {
  public:
    yakl::SArray<T,1,MAX_FIELDS> fields;
    int num_fields;

    KOKKOS_INLINE_FUNCTION MultipleFields() { num_fields = 0; }

    KOKKOS_INLINE_FUNCTION MultipleFields(MultipleFields const &rhs) {
      this->num_fields = rhs.num_fields;
      for (int i=0; i < num_fields; i++) {
        this->fields(i) = rhs.fields(i);
      }
    }

    KOKKOS_INLINE_FUNCTION MultipleFields & operator=(MultipleFields const &rhs) {
      this->num_fields = rhs.num_fields;
      for (int i=0; i < num_fields; i++) {
        this->fields(i) = rhs.fields(i);
      }
      return *this;
    }

    KOKKOS_INLINE_FUNCTION MultipleFields(MultipleFields &&rhs) {
      this->num_fields = rhs.num_fields;
      for (int i=0; i < num_fields; i++) {
        this->fields(i) = rhs.fields(i);
      }
    }

    KOKKOS_INLINE_FUNCTION MultipleFields& operator=(MultipleFields &&rhs) {
      this->num_fields = rhs.num_fields;
      for (int i=0; i < num_fields; i++) {
        this->fields(i) = rhs.fields(i);
      }
      return *this;
    }

    KOKKOS_INLINE_FUNCTION void add_field( T field ) {
      this->fields(num_fields) = field;
      num_fields++;
    }

    KOKKOS_INLINE_FUNCTION T &get_field( int tr ) const {
      return this->fields(tr);
    }

    KOKKOS_INLINE_FUNCTION int get_num_fields() const { return num_fields; }
    KOKKOS_INLINE_FUNCTION int size          () const { return num_fields; }

    KOKKOS_INLINE_FUNCTION auto operator() (int tr, int i1) const ->
                                decltype(fields(tr)(i1)) {
      return this->fields(tr)(i1);
    }
    KOKKOS_INLINE_FUNCTION auto operator() (int tr, int i1, int i2) const ->
                                decltype(fields(tr)(i1,i2)) {
      return this->fields(tr)(i1,i2);
    }
    KOKKOS_INLINE_FUNCTION auto operator() (int tr, int i1, int i2, int i3) const ->
                                decltype(fields(tr)(i1,i2,i3)) {
      return this->fields(tr)(i1,i2,i3);
    }
    KOKKOS_INLINE_FUNCTION auto operator() (int tr, int i1, int i2, int i3, int i4) const ->
                                decltype(fields(tr)(i1,i2,i3,i4)) {
      return this->fields(tr)(i1,i2,i3,i4);
    }
    KOKKOS_INLINE_FUNCTION auto operator() (int tr, int i1, int i2, int i3, int i4, int i5) const ->
                                decltype(fields(tr)(i1,i2,i3,i4,i5)) {
      return this->fields(tr)(i1,i2,i3,i4,i5);
    }
    KOKKOS_INLINE_FUNCTION auto operator() (int tr, int i1, int i2, int i3, int i4, int i5, int i6) const ->
                                decltype(fields(tr)(i1,i2,i3,i4,i5,i6)) {
      return this->fields(tr)(i1,i2,i3,i4,i5,i6);
    }
    KOKKOS_INLINE_FUNCTION auto operator() (int tr, int i1, int i2, int i3, int i4, int i5, int i6, int i7) const ->
                                decltype(fields(tr)(i1,i2,i3,i4,i5,i6,i7)) {
      return this->fields(tr)(i1,i2,i3,i4,i5,i6,i7);
    }
    KOKKOS_INLINE_FUNCTION auto operator() (int tr, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8) const ->
                                decltype(fields(tr)(i1,i2,i3,i4,i5,i6,i7,i8)) {
      return this->fields(tr)(i1,i2,i3,i4,i5,i6,i7,i8);
    }
  };



  template <class T, int N>
  using MultiField = MultipleFields< max_fields , Array<T,N,memDevice,styleC> >;
}


