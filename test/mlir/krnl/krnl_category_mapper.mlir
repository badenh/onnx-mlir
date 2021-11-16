// RUN: onnx-mlir-opt --convert-krnl-to-affine --convert-krnl-to-llvm %s -split-input-file | FileCheck %s

// -----

// Test that we can receive and return an array of strings 
func private @test_string(%arg0 : memref<2x!krnl.string>) -> memref<2x!krnl.string>  {
  return %arg0 : memref<2x!krnl.string>

  // CHECK-LABEL: test_string
  // CHECK: [[DESC:%.+]] = llvm.mlir.undef : !llvm.struct<(ptr<struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>>, ptr<struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: [[INS1:%.+]] = llvm.insertvalue %arg0, [[DESC]][0] : !llvm.struct<(ptr<struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>>, ptr<struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: [[INS2:%.+]] = llvm.insertvalue %arg1, [[INS1]][1] : !llvm.struct<(ptr<struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>>, ptr<struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: [[INS3:%.+]] = llvm.insertvalue %arg2, [[INS2]][2] : !llvm.struct<(ptr<struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>>, ptr<struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: [[INS4:%.+]] = llvm.insertvalue %arg3, [[INS3]][3, 0] : !llvm.struct<(ptr<struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>>, ptr<struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: [[INS5:%.+]] = llvm.insertvalue %arg4, [[INS4]][4, 0] : !llvm.struct<(ptr<struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>>, ptr<struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK:                llvm.return [[INS5]] : !llvm.struct<(ptr<struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>>, ptr<struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>>, i64, array<1 x i64>, array<1 x i64>)>
}

// -----

// Test that 'krnl.global' for an array of strings is lowerer correctly.
func private @test_krnl_global_for_string() -> memref<2x!krnl.string>  {
  %0 = "krnl.global"() {name = "cat_strings", shape = [2], value = dense<["cat", "dog"]> : tensor<2x!krnl.string>} : () -> memref<2x!krnl.string>  
  return %0 : memref<2x!krnl.string>  

  // CHECK-DAG: llvm.mlir.global internal constant @cat("cat")
  // CHECK-DAG: llvm.mlir.global internal constant @dog("dog")

  // CHECK-LABEL: @test_krnl_global_for_string
  // CHECK: [[C2:%.+]]       = llvm.mlir.constant(2 : i32) : i32
  // CHECK: [[ALLOCA:%.+]]   = llvm.alloca [[C2]] x !llvm.ptr<i8> {alignment = 16 : i64} : (i32) -> !llvm.ptr<ptr<i8>>
  // CHECK: [[CAT_ADDR:%.+]] = llvm.mlir.addressof @cat : !llvm.ptr<array<3 x i8>>
  // CHECK: [[C0:%.+]]       = llvm.mlir.constant(0 : index) : i64
  // CHECK: [[PCAT:%.+]]     = llvm.getelementptr [[CAT_ADDR]]{{.*}}[[C0]], [[C0]]{{.*}} : (!llvm.ptr<array<3 x i8>>, i64, i64) -> !llvm.ptr<i8>
  // CHECK: [[C01:%.+]]      = llvm.mlir.constant(0 : index) : i64
  // CHECK: [[PTR1:%.+]]     = llvm.getelementptr [[ALLOCA]]{{.*}}[[C01]]{{.*}} : (!llvm.ptr<ptr<i8>>, i64) -> !llvm.ptr<ptr<i8>>
  // CHECK:                    llvm.store [[PCAT]], [[PTR1]] : !llvm.ptr<ptr<i8>>
  // CHECK: [[DOG_ADDR:%.+]] = llvm.mlir.addressof @dog : !llvm.ptr<array<3 x i8>>
  // CHECK: [[C02:%.+]]      = llvm.mlir.constant(0 : index) : i64
  // CHECK: [[PDOG:%.+]]     = llvm.getelementptr [[DOG_ADDR]]{{.*}}[[C02]], [[C02]]{{.*}} : (!llvm.ptr<array<3 x i8>>, i64, i64) -> !llvm.ptr<i8>
  // CHECK: [[C1:%.+]]       = llvm.mlir.constant(1 : index) : i64
  // CHECK: [[PTR2:%.+]]     = llvm.getelementptr [[ALLOCA]]{{.*}}[[C1]]{{.*}} : (!llvm.ptr<ptr<i8>>, i64) -> !llvm.ptr<ptr<i8>>
  // CHECK:                    llvm.store [[PDOG]], [[PTR2]] : !llvm.ptr<ptr<i8>>
}

// -----

// Test that 'krnl.strlen' is lowered to 'strlen' when the argument is passed into the function.
func private @test_strlen1(%arg0: !krnl.string) -> i64  {
  %len = "krnl.strlen"(%arg0) : (!krnl.string) -> i64
  return %len : i64

  // CHECK: llvm.func @strlen(!llvm.ptr<i8>) -> i64
  // CHECK-LABEL: @test_strlen1
  // CHECK: [[STR:%.+]] = llvm.extractvalue %arg0[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: [[LEN:%.+]] = llvm.call @strlen([[STR]]) : (!llvm.ptr<i8>) -> i64
  // CHECK:               llvm.return [[LEN]] : i64
}

// -----

// Test that 'krnl.strlen' is lowered to 'strlen' when the argument is created in the function.
func private @test_strlen2() -> i64  {
  %c0 = arith.constant 0 : index  
  %ptr_str = memref.alloc() {alignment = 16 : i64} : memref<1x!krnl.string>
  %str = krnl.load %ptr_str[%c0] : memref<1x!krnl.string>
  %len = "krnl.strlen"(%str) : (!krnl.string) -> i64
  return %len : i64

  // CHECK: llvm.func @strlen(!llvm.ptr<i8>) -> i64
  // CHECK-LABEL: @test_strlen2
  // CHECK: [[STR:%.+]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: [[LEN:%.+]] = llvm.call @strlen([[STR]]) : (!llvm.ptr<i8>) -> i64
  // CHECK:               llvm.return [[LEN]] : i64
}

// -----

// Test that 'krnl.strncmp' is lowered to a call to the 'strncmp' standard C function.
func private @test_strncmp(%str: !krnl.string, %len: i64) -> i32  {
  %c0 = arith.constant 0 : index
  %ptr = memref.alloc() {alignment = 16 : i64} : memref<1x!krnl.string>
  %str1 = krnl.load %ptr[%c0] : memref<1x!krnl.string>
  %cmp = "krnl.strncmp"(%str, %str1, %len) : (!krnl.string, !krnl.string, i64) -> i32
  return %cmp : i32

  // CHECK: llvm.func @strncmp(!llvm.ptr<i8>, !llvm.ptr<i8>, i64) -> i32  
  // CHECK-LABEL: @test_strncmp(%arg0: !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>, %arg1: i64)
  // CHECK: [[LOAD:%.+]] = llvm.load {{.*}} : !llvm.ptr<struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>>
  // CHECK: [[STR1:%.+]] = llvm.extractvalue %arg0[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: [[STR2:%.+]] = llvm.extractvalue %26[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: [[CMP:%.+]]  = llvm.call @strncmp([[STR1]], [[STR2]], %arg1) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64) -> i32
  // CHECK:                llvm.return [[CMP]] : i32
}

// -----

// Test that 'krnl.find_index' can be called when the first argument is a string.
func private @test_find_index_str(%str: !krnl.string) -> index {
  %G = "krnl.global"() {name = "G", shape = [3], value = dense<[1,0,-3]> : vector<3xi32>} : () -> memref<3xi32>
  %V = "krnl.global"() {name = "V", shape = [3], value = dense<[1,2,0]> : vector<3xi32>} : () -> memref<3xi32>
  %c3 = arith.constant 3 : i32  
  %index = "krnl.find_index"(%str, %G, %V, %c3) : (!krnl.string, memref<3xi32>, memref<3xi32>, i32) -> index
  return %index : index

  // CHECK-DAG: llvm.func @find_index_str(!llvm.ptr<i8>, !llvm.ptr<i32>, !llvm.ptr<i32>, i32) -> i32
  // CHECK-DAG: llvm.mlir.global internal constant @V(dense<[1, 2, 0]> : vector<3xi32>) : !llvm.array<3 x i32>
  // CHECK-DAG: llvm.mlir.global internal constant @G(dense<[1, 0, -3]> : vector<3xi32>) : !llvm.array<3 x i32>  

  // CHECK-LABEL: @test_find_index_str(%arg0: !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>)
  // CHECK: [[GADDR:%.+]] = llvm.mlir.addressof @G : !llvm.ptr<array<3 x i32>>  
  // CHECK: [[C1:%.+]]    = llvm.mlir.constant(1 : index) : i64  
  // CHECK: [[INS1:%.+]]  = llvm.insertvalue [[C1]], {{.*}}[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>  
  // CHECK: [[VADDR:%.+]] = llvm.mlir.addressof @V : !llvm.ptr<array<3 x i32>>
  // CHECK: [[C11:%.+]]   = llvm.mlir.constant(1 : index) : i64
  // CHECK: [[INS2:%.+]]  = llvm.insertvalue [[C11]], {{.*}}[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: [[LEN:%.+]]   = llvm.mlir.constant(3 : i32) : i32
  // CHECK: [[STR:%.+]]   = llvm.extractvalue %arg0[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: [[G:%.+]]     = llvm.extractvalue [[INS1]][1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: [[V:%.+]]     = llvm.extractvalue [[INS2]][1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: [[INDEX:%.+]] = llvm.call @find_index_str([[STR]], [[G]], [[V]], [[LEN]]) : (!llvm.ptr<i8>, !llvm.ptr<i32>, !llvm.ptr<i32>, i32) -> i32
  // CHECK: [[RET:%.+]]   = builtin.unrealized_conversion_cast [[INDEX]] : i32 to i64
  // CHECK:                 llvm.return [[RET]] : i64
}

// -----

// Test that 'krnl.find_index' can be called when the first argument is a int64_t.
func private @test_find_index_int(%val: i64) -> index {
  %G = "krnl.global"() {name = "G", shape = [3], value = dense<[1,0,-3]> : vector<3xi32>} : () -> memref<3xi32>
  %V = "krnl.global"() {name = "V", shape = [3], value = dense<[1,2,0]> : vector<3xi32>} : () -> memref<3xi32>
  %c3 = arith.constant 3 : i32  
  %index = "krnl.find_index"(%val, %G, %V, %c3) : (i64, memref<3xi32>, memref<3xi32>, i32) -> index
  return %index : index

  // CHECK-DAG: llvm.func @find_index_i64(i64, !llvm.ptr<i32>, !llvm.ptr<i32>, i32) -> i32
  // CHECK-DAG: llvm.mlir.global internal constant @V(dense<[1, 2, 0]> : vector<3xi32>) : !llvm.array<3 x i32>
  // CHECK-DAG: llvm.mlir.global internal constant @G(dense<[1, 0, -3]> : vector<3xi32>) : !llvm.array<3 x i32>

  // CHECK-LABEL: @test_find_index_int(%arg0: i64)
  // CHECK: [[GADDR:%.+]] = llvm.mlir.addressof @G : !llvm.ptr<array<3 x i32>>  
  // CHECK: [[C1:%.+]]    = llvm.mlir.constant(1 : index) : i64  
  // CHECK: [[INS1:%.+]]  = llvm.insertvalue [[C1]], {{.*}}[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>  
  // CHECK: [[VADDR:%.+]] = llvm.mlir.addressof @V : !llvm.ptr<array<3 x i32>>
  // CHECK: [[C11:%.+]]   = llvm.mlir.constant(1 : index) : i64
  // CHECK: [[INS2:%.+]]  = llvm.insertvalue [[C11]], {{.*}}[4, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: [[LEN:%.+]]   = llvm.mlir.constant(3 : i32) : i32
  // CHECK: [[G:%.+]]     = llvm.extractvalue [[INS1]][1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: [[V:%.+]]     = llvm.extractvalue [[INS2]][1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: [[INDEX:%.+]] = llvm.call @find_index_i64(%arg0, [[G]], [[V]], [[LEN]]) : (i64, !llvm.ptr<i32>, !llvm.ptr<i32>, i32) -> i32
  // CHECK: [[RET:%.+]]   = builtin.unrealized_conversion_cast [[INDEX]] : i32 to i64
  // CHECK:                 llvm.return [[RET]] : i64
}