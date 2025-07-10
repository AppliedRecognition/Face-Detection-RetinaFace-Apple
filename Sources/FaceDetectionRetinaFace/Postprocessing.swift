//
//  Postprocessing.swift
//
//
//  Created by Jakub Dolejs on 07/07/2025.
//

import Foundation
import CoreGraphics
import VerIDCommonTypes
import Accelerate
import CoreML

struct Postprocessing {
    
    func decodeBoxes(_ boxes: MLMultiArray, scores: MLMultiArray, landmarks: MLMultiArray) throws -> [DetectionBox] {
        let rects: [CGRect]
        let confidenceScores: [Float]
        let lms: [[Float]]
        let priors = self.generatePriors()
        let scoreCount = priors[0].count * 2
        let boxCount = priors[0].count * 4
        let landmarkCount = priors[0].count * 10
        if #available(iOS 15.4, *) {
            rects = try boxes.withUnsafeBufferPointer(ofType: Float.self) {
                return try decodeBoxesFromPointer($0.baseAddress!, priors: priors)
            }
            confidenceScores = scores.withUnsafeBufferPointer(ofType: Float.self) {
                self.readValuesFromPointer($0.baseAddress!, startIndex: 1, stride: 2, count: scoreCount)
            }
            lms = landmarks.withUnsafeBufferPointer(ofType: Float.self) {
                self.decodeLandmarksFromPointer($0.baseAddress!, priors: priors)
            }
        } else {
            rects = try decodeBoxesFromPointer(boxes.dataPointer.bindMemory(to: Float.self, capacity: boxCount), priors: priors)
            confidenceScores = self.readValuesFromPointer(scores.dataPointer.bindMemory(to: Float.self, capacity: scoreCount), startIndex: 1, stride: 2, count: scores.count)
            lms = self.decodeLandmarksFromPointer(landmarks.dataPointer.bindMemory(to: Float.self, capacity: landmarkCount), priors: priors)
        }
        let detections: [DetectionBox] = zip(rects, confidenceScores).enumerated().map({ (index, rectScore) in
            let landmarks: [CGPoint] = (0..<5).map { (i: Int) in
                let x = CGFloat(lms[i * 2][index])
                let y = CGFloat(lms[i * 2 + 1][index])
                return CGPoint(x: x, y: y)
            }
            return DetectionBox(score: rectScore.1, bounds: rectScore.0, landmarks: landmarks, angle: EulerAngle(), quality: rectScore.1 * 10)
        })
        return detections
    }
    
    func nonMaxSuppression(boxes: [DetectionBox], iouThreshold: Float, limit: Int) -> [DetectionBox] {
        var selected: [DetectionBox] = []
        let sorted = boxes.sorted(by: { $0.score > $1.score })
        
        for box in sorted {
            if selected.count >= limit { break }
            if selected.allSatisfy({ iou($0.bounds, box.bounds) < CGFloat(iouThreshold) }) {
                selected.append(box)
            }
        }
        
        return selected
    }
    
    func calculateFaceAngle(leftEye: CGPoint, rightEye: CGPoint, noseTip: CGPoint, leftMouth: CGPoint, rightMouth: CGPoint) -> EulerAngle<Float> {
        let dx = rightEye.x - leftEye.x
        let dy = rightEye.y - leftEye.y
        let roll = atan2(dy, dx).degrees
        
        let eyeCenter = CGPoint(x: (leftEye.x + rightEye.x) / 2, y: (leftEye.y + rightEye.y) / 2)
        let mouthCenter = CGPoint(x: (leftMouth.x + rightMouth.x) / 2, y: (leftMouth.y + rightMouth.y) / 2)
        
        let interocular = rightEye.x - leftEye.x
        let noseOffset = noseTip.x - eyeCenter.x
        let yaw = atan2(noseOffset, interocular).degrees * 1.2  // Tweak factor
        
        let verticalFaceLength = mouthCenter.y - eyeCenter.y
        let verticalNoseOffset = noseTip.y - eyeCenter.y
        let pitchRatio = verticalNoseOffset / verticalFaceLength
        let pitch = (0.5 - pitchRatio) * 90  // Neutral face at pitch ≈ 0
        
        return EulerAngle(yaw: yaw.asFloat, pitch: pitch.asFloat, roll: roll.asFloat)
    }
    
    private func readValuesFromPointer(_ pointer: UnsafePointer<Float>, startIndex: Int, stride: Int, count: Int) -> [Float] {
        let indices = Swift.stride(from: startIndex, to: count, by: stride).map({ Float($0) })
        let n = indices.count
        let result = [Float](unsafeUninitializedCapacity: n) { resultBuffer, initializedCount in
            vDSP_vindex(pointer, indices, 1, resultBuffer.baseAddress!, 1, vDSP_Length(n))
            initializedCount = n
        }
        return result
    }
    
    private func decodeLandmarksFromPointer(_ pointer: UnsafePointer<Float>, priors: [[Float]]) -> [[Float]] {
        let count = priors[0].count
        let predictions: [[Float]] = (0..<10).map { i in
            readValuesFromPointer(pointer, startIndex: i, stride: 10, count: count * 10)
        }
        let variance: Float = 0.1
        var landmarks: [[Float]] = [[Float]](repeating: [Float](repeating: .zero, count: count), count: 10)
        for i in 0..<5 {
            // X component
            let dx = predictions[2 * i]
            var tempW = [Float](repeating: 0, count: count)
            var outX = [Float](repeating: 0, count: count)
            
            vDSP.multiply(variance, priors[2], result: &tempW)
            vDSP.multiply(dx, tempW, result: &tempW)
            vDSP.add(priors[0], tempW, result: &outX)
            landmarks[2 * i] = outX
            
            // Y component
            let dy = predictions[2 * i + 1]
            var tempH = [Float](repeating: 0, count: count)
            var outY = [Float](repeating: 0, count: count)
            
            vDSP.multiply(variance, priors[3], result: &tempH)
            vDSP.multiply(dy, tempH, result: &tempH)
            vDSP.add(priors[1], tempH, result: &outY)
            landmarks[2 * i + 1] = outY
        }
        
        return landmarks
    }
    
    private func decodeBoxesFromPointer(_ pointer: UnsafePointer<Float>, priors: [[Float]]) throws -> [CGRect] {
        let count = priors[0].count
        let boxCount = count * 4
        let dx = readValuesFromPointer(pointer, startIndex: 0, stride: 4, count: boxCount)
        let dy = readValuesFromPointer(pointer, startIndex: 1, stride: 4, count: boxCount)
        let dw = readValuesFromPointer(pointer, startIndex: 2, stride: 4, count: boxCount)
        let dh = readValuesFromPointer(pointer, startIndex: 3, stride: 4, count: boxCount)
        let cx = priors[0]
        let cy = priors[1]
        let pw = priors[2]
        let ph = priors[3]
        let variance0: Float = 0.1
        
        var adjX = [Float](repeating: 0, count: count)
        var adjY = [Float](repeating: 0, count: count)
        
        // dx * variance * pw → temp
        vDSP.multiply(dx, pw, result: &adjX)
        vDSP.multiply(variance0, adjX, result: &adjX)
        
        // dy * variance * ph
        vDSP.multiply(dy, ph, result: &adjY)
        vDSP.multiply(variance0, adjY, result: &adjY)
        
        // cx + adjusted dx
        vDSP.add(cx, adjX, result: &adjX)
        vDSP.add(cy, adjY, result: &adjY)
        
        let variance1: Float = 0.2
        
        var expW = [Float](repeating: 0, count: count)
        var expH = [Float](repeating: 0, count: count)
        
        vDSP.multiply(variance1, dw, result: &expW)
        vDSP.multiply(variance1, dh, result: &expH)
        
        vvexpf(&expW, expW, [Int32(count)])
        vvexpf(&expH, expH, [Int32(count)])
        
        vDSP.multiply(pw, expW, result: &expW)
        vDSP.multiply(ph, expH, result: &expH)
        
        var x1 = [Float](repeating: 0, count: count)
        var y1 = [Float](repeating: 0, count: count)
        var x2 = [Float](repeating: 0, count: count)
        var y2 = [Float](repeating: 0, count: count)
        
        var divisor: Float = 2
        var halfW = [Float](repeating: 0, count: count)
        var halfH = [Float](repeating: 0, count: count)
        vDSP_vsdiv(expW, 1, &divisor, &halfW, 1, vDSP_Length(expW.count))
        vDSP_vsdiv(expH, 1, &divisor, &halfH, 1, vDSP_Length(expH.count))
        vDSP.subtract(adjX, halfW, result: &x1)
        vDSP.subtract(adjY, halfH, result: &y1)
        vDSP.add(adjX, halfW, result: &x2)
        vDSP.add(adjY, halfH, result: &y2)
        
        return (0..<(count)).map { (i: Int) in
            CGRect(x: CGFloat(x1[i]), y: CGFloat(y1[i]), width: CGFloat(x2[i] - x1[i]), height: CGFloat(y2[i] - y1[i]))
        }
    }
    
    private func generatePriors() -> [[Float]] {
        return Priors(
            minSizes: [[16, 32], [64, 128], [256, 512]],
            steps: [8, 16, 32],
            clip: false,
            imageWidth: 640,
            imageHeight: 640
        ).generate()
    }
    
    private func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let inter = a.intersection(b)
        let interArea = inter.width * inter.height
        guard interArea > 0, interArea > CGFloat.ulpOfOne else { return 0 }
        let union = a.width * a.height + b.width * b.height - interArea
        return interArea / union
    }
}

struct DetectionBox: Encodable {
    let score: Float
    let bounds: CGRect
    let landmarks: [CGPoint]
    let angle: EulerAngle<Float>
    let quality: Float
    
    enum CodingKeys: CodingKey {
        case bounds, landmarks
    }
    
    enum BoundsCodingKeys: CodingKey {
        case x, y, width, height
    }
    
    func applyingTransform(_ transform: CGAffineTransform) -> DetectionBox {
        return DetectionBox(score: self.score, bounds: self.bounds.applying(transform), landmarks: self.landmarks.map { $0.applying(transform) }, angle: self.angle, quality: self.quality)
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        var boundsContainer = container.nestedContainer(keyedBy: BoundsCodingKeys.self, forKey: .bounds)
        try boundsContainer.encode(self.bounds.minX, forKey: .x)
        try boundsContainer.encode(self.bounds.minY, forKey: .y)
        try boundsContainer.encode(self.bounds.width, forKey: .width)
        try boundsContainer.encode(self.bounds.height, forKey: .height)
        try container.encode(self.landmarks.map({ EncodablePoint(point: $0) }), forKey: .landmarks)
    }
}

fileprivate struct EncodablePoint: Encodable {
    let x: CGFloat
    let y: CGFloat
    init(point: CGPoint) {
        self.x = point.x
        self.y = point.y
    }
}

fileprivate struct Priors {
    let minSizes: [[Int]]
    let steps: [Int]
    let clip: Bool
    let imageWidth: Int
    let imageHeight: Int
    
    func generate() -> [[Float]] {
        var anchors: [[Float]] = [
            [Float](),[Float](),[Float](),[Float]()
        ]
        
        for (k, step) in steps.enumerated() {
            let minSizesForStep = minSizes[k]
            let featureMapHeight = Int(ceil(Float(imageHeight) / Float(step)))
            let featureMapWidth = Int(ceil(Float(imageWidth) / Float(step)))
            
            for i in 0..<featureMapHeight {
                for j in 0..<featureMapWidth {
                    for minSize in minSizesForStep {
                        let s_kx = Float(minSize) / Float(imageWidth)
                        let s_ky = Float(minSize) / Float(imageHeight)
                        let cx = (Float(j) + 0.5) * Float(step) / Float(imageWidth)
                        let cy = (Float(i) + 0.5) * Float(step) / Float(imageHeight)
                        
                        var anchor: [Float] = [cx, cy, s_kx, s_ky]
                        if clip {
                            anchor = anchor.map { max(0, min(1, $0)) }
                        }
                        anchors[0].append(cx)
                        anchors[1].append(cy)
                        anchors[2].append(s_kx)
                        anchors[3].append(s_ky)
                    }
                }
            }
        }
        
        return anchors
    }
}

fileprivate extension CGFloat {
    
    var asFloat: Float {
        return Float(self)
    }
    
    var degrees: CGFloat {
        return self * 180 / .pi
    }
}

fileprivate extension Float {
    
    var degrees: Float {
        return self * 180 / .pi
    }
}
