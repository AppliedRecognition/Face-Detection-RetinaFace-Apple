//
//  Preprocessing.swift
//
//
//  Created by Jakub Dolejs on 07/07/2025.
//

import UIKit
import CoreML
import Accelerate

struct Preprocessing {
    
    func scalePixelBuffer(_ pixelBuffer: CVPixelBuffer, to size: CGSize) throws -> (CVPixelBuffer, CGFloat) {
        let sourceImage = CIImage(cvPixelBuffer: pixelBuffer)
        let sourceWidth = sourceImage.extent.width
        let sourceHeight = sourceImage.extent.height
        
        // Calculate aspect-fit scale
        let scale = min(size.width / sourceWidth, size.height / sourceHeight)
        let scaledWidth = sourceWidth * scale
        let scaledHeight = sourceHeight * scale
        
        // Scale and align to top-left (no offset)
        let scaleTransform = CGAffineTransform(scaleX: scale, y: scale)
        let resizedImage = sourceImage.transformed(by: scaleTransform)
        
        // Create output pixel buffer
        var outputBuffer: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ] as CFDictionary
        
        CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(size.width),
            Int(size.height),
            kCVPixelFormatType_32BGRA,
            attrs,
            &outputBuffer
        )
        
        guard let paddedBuffer = outputBuffer else {
            throw FaceDetectionRetinaFaceError.imageResizingError
        }
        
        // Clear unused area (background = black)
        CVPixelBufferLockBaseAddress(paddedBuffer, [])
        let base = CVPixelBufferGetBaseAddress(paddedBuffer)!
        defer {
            CVPixelBufferUnlockBaseAddress(paddedBuffer, [])
        }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(paddedBuffer)
        memset(base, 0, bytesPerRow * Int(size.height))
        
        // Render resized image into top-left of output buffer
        let drawRect = CGRect(origin: .zero, size: CGSize(width: scaledWidth, height: scaledHeight))
        CIContext().render(resizedImage, to: paddedBuffer, bounds: drawRect, colorSpace: CGColorSpaceCreateDeviceRGB())
        
        return (paddedBuffer, scale)
    }
}
