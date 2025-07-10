# RetinaFace Face Detection

[Face detection](https://appliedrecognition.github.io/Ver-ID-Common-Types-Apple/documentation/veridcommontypes/facedetection) implementation for Ver-ID SDK using RetinaFace model.

| Platform | Min. version | Distribution |
| --- | --- | --- |
| iOS | 15 | [SPM](https://www.swift.org/documentation/package-manager/) |

## Installation

The library can be installed using [Swift Package Manager](https://www.swift.org/documentation/package-manager/):

1. Open your project in **Xcode**.
2. Click on the **Package Dependencies** tab.
3. Click the **+ button** to add a new package.
4. Enter `https://github.com/AppliedRecognition/Face-Detection-RetinaFace-Apple.git` in the search bar.
5. In the **Dependency Rule** drop-down menu select **Up to Next Major Version** and in the adjacent text field enter `1.0.0`.
6. Click the **Add Package** button.

## Usage

```swift
import UIKit
import FaceDetectionRetinaFace

// Detect a face in image
func detectFaceInImage(_ image: CGImage) async throws -> Face? {
    let detection = try FaceDetectionRetinaFace()
    guard let verIDImage = Image(cgImage: image) else {
        throw NSError(domain: "FaceDetection", code: 1)
    }
    let faces = try await detection.detectFacesInImage(verIDImage, limit: 1)
    return faces.first
}

// Return an image with the face and facial landmarks drawn on top
func drawFace(_ face: Face, onImage image: CGImage) -> UIImage {
    let format = UIGraphicsImageRendererFormat()
    format.scale = 1.0
    let annotatedImage = UIGraphicsImageRenderer(size: CGSize(width: image.width, height: image.height), format: format).image { context in
        // Paste the original image
        UIImage(cgImage: image).draw(at: .zero)
        UIColor.green.setStroke()
        // Draw the face outline
        let path = UIBezierPath(rect: face.bounds)
        path.lineWidth = 10
        path.stroke()
        UIColor.green.setFill()
        let dotRadius: CGFloat = 5
        // Draw the face landmarks
        for landmark in face.landmarks {
            UIBezierPath(arcCenter: landmark, radius: dotRadius, startAngle: 0, endAngle: .pi * 2, clockwise: true).fill()
        }
    }
    return annotatedImage
}
```