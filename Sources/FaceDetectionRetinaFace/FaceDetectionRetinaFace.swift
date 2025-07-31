// The Swift Programming Language
// https://docs.swift.org/swift-book

import Foundation
import CoreML
import Vision
import Accelerate
import VerIDCommonTypes
import UIKit

/// Face detection implementation using RetinaFace model
public final class FaceDetectionRetinaFace: FaceDetection {
    private let model: MLModel
    private let inputSize = CGSize(width: 320, height: 320)
    private let modelInputPrep = Preprocessing()
    private let postProcessing = Postprocessing()
    
    /// Initializer
    /// - Throws: Exception of ML model initialization fails
    public init() throws {
        let config = MLModelConfiguration()
        config.allowLowPrecisionAccumulationOnGPU = true
        self.model = try RetinaFace(configuration: config).model
    }
    
    /// Detect faces in image
    /// - Parameters:
    ///   - image: Image in which to detect faces
    ///   - limit: Maximum number of faces to detect
    /// - Returns: Array of detected faces
    public func detectFacesInImage(_ image: Image, limit: Int) async throws -> [Face] {
        let (resizedPixelBuffer, scale) = try self.modelInputPrep.scalePixelBuffer(image.videoBuffer, to: self.inputSize)
        let input = RetinaFaceInput(input: resizedPixelBuffer)
        let prediction = try model.prediction(from: input)
        
        // Extract outputs - adjust keys based on actual model
        guard
            let scoresMultiArray = prediction.featureValue(for: "scores")?.multiArrayValue,
            let boxesMultiArray = prediction.featureValue(for: "boxes")?.multiArrayValue,
            let landmarksMultiArray = prediction.featureValue(for: "landmarks")?.multiArrayValue
        else {
            throw FaceDetectionRetinaFaceError.missingExpectedModelOutputs
        }
        var boxes = try self.postProcessing.decodeBoxes(boxesMultiArray, scores: scoresMultiArray, landmarks: landmarksMultiArray)
        boxes = self.postProcessing.nonMaxSuppression(boxes: boxes, iouThreshold: 0.4, limit: limit)
        let transform = CGAffineTransform(scaleX: inputSize.width, y: inputSize.height).concatenating(CGAffineTransform(scaleX: 1 / scale, y: 1 / scale))
        boxes = boxes.map { $0.applyingTransform(transform) }
        return boxes.map {
            let angle = postProcessing.calculateFaceAngle(leftEye: $0.landmarks[0], rightEye: $0.landmarks[1], noseTip: $0.landmarks[2], leftMouth: $0.landmarks[3], rightMouth: $0.landmarks[4])
            return Face(
                bounds: $0.bounds,
                angle: angle,
                quality: $0.quality,
                landmarks: $0.landmarks,
                leftEye: $0.landmarks[0],
                rightEye: $0.landmarks[1],
                noseTip: $0.landmarks[2],
                mouthLeftCorner: $0.landmarks[3],
                mouthRightCorner: $0.landmarks[4]
            )
        }
    }
}
