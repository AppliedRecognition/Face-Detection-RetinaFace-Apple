// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "FaceDetectionRetinaFace",
    platforms: [.iOS(.v15)],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "FaceDetectionRetinaFace",
            targets: ["FaceDetectionRetinaFace"]),
    ],
    dependencies: [
        .package(url: "https://github.com/AppliedRecognition/Ver-ID-Common-Types-Apple.git", .upToNextMajor(from: "2.1.0"))
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
            name: "FaceDetectionRetinaFace",
            dependencies: [
                .product(name: "VerIDCommonTypes", package: "Ver-ID-Common-Types-Apple")
            ],
            resources: [
                .process("Resources")
            ]),
        .testTarget(
            name: "FaceDetectionRetinaFaceTests",
            dependencies: [
                "FaceDetectionRetinaFace",
                .product(name: "VerIDCommonTypes", package: "Ver-ID-Common-Types-Apple")
            ],
            resources: [
                .process("Resources")
            ]),
    ]
)
