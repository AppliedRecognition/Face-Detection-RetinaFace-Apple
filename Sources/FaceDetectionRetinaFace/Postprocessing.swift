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
    
    let boxIndices: [[Int]]
    let scoreIndices: [[Float]]
    let landmarkIndices: [[Int]]
    let priors: [[Float]]
    let scoreThreshold: Float = 0.3
    
    init() {
        let predictionCount = 16800
        let boxCount = 4
        boxIndices = (0..<boxCount).map { i in
            Array(stride(from: i, to: predictionCount * boxCount, by: boxCount))
        }
        let scoreCount = 2
        scoreIndices = (0..<scoreCount).map { i in
            stride(from: i, to: predictionCount * scoreCount, by: scoreCount).map { Float($0) }
        }
        let landmarkCount = 10
        landmarkIndices = (0..<landmarkCount).map { i in
            Array(stride(from: i, to: predictionCount * landmarkCount, by: landmarkCount))
        }
        priors = Priors(
            minSizes: [[16, 32], [64, 128], [256, 512]],
            steps: [8, 16, 32],
            clip: false,
            imageWidth: 640,
            imageHeight: 640
        ).generate()
    }
    
    func decodeBoxes(_ boxes: MLMultiArray, scores: MLMultiArray, landmarks: MLMultiArray) throws -> [DetectionBox] {
        let boxesArray = self.extractValuesFromMLMultiArray(boxes)
        let scoresArray = self.extractValuesFromMLMultiArray(scores)
        let landmarkArray = self.extractValuesFromMLMultiArray(landmarks)
        var confScores = [Float](repeating: 0, count: scoresArray.count / 2)
        vDSP_vindex(scoresArray, scoreIndices[1], 1, &confScores, 1, vDSP_Length(confScores.count))
        let retainedIndices = (0..<confScores.count).filter { confScores[$0] >= scoreThreshold }
        if retainedIndices.isEmpty {
            return []
        }
        let cx = priors[0]
        let cy = priors[1]
        let pw = priors[2]
        let ph = priors[3]
        
        var detections: [DetectionBox] = []
        for idx in retainedIndices {
            
            // -- Box decoding --
            let dx = boxesArray[boxIndices[0][idx]]
            let dy = boxesArray[boxIndices[1][idx]]
            let dw = boxesArray[boxIndices[2][idx]]
            let dh = boxesArray[boxIndices[3][idx]]
            
            let adjX = cx[idx] + 0.1 * dx * pw[idx]
            let adjY = cy[idx] + 0.1 * dy * ph[idx]
            let expW = pw[idx] * exp(0.2 * dw)
            let expH = ph[idx] * exp(0.2 * dh)
            
            let x1 = adjX - expW / 2
            let y1 = adjY - expH / 2
            let rect = CGRect(x: CGFloat(x1), y: CGFloat(y1), width: CGFloat(expW), height: CGFloat(expH))
            
            // -- Landmark decoding --
            var landmarks: [CGPoint] = []
            for i in 0..<5 {
                let lx = landmarkArray[landmarkIndices[2*i][idx]]
                let ly = landmarkArray[landmarkIndices[2*i+1][idx]]
                
                let pointX = cx[idx] + 0.1 * lx * pw[idx]
                let pointY = cy[idx] + 0.1 * ly * ph[idx]
                landmarks.append(CGPoint(x: CGFloat(pointX), y: CGFloat(pointY)))
            }
            
            detections.append(DetectionBox(
                score: confScores[idx],
                bounds: rect,
                landmarks: landmarks,
                angle: EulerAngle(),
                quality: confScores[idx]
            ))
        }
        return detections
    }
    
//    private func decodeBoxesFromArray(_ array: [Float], at indices: [Int]) -> [CGRect] {
//        let count = priors[0].count
//        precondition(array.count == count * 4, "Input array size mismatch")
//        
//        // Split input array: dx, dy, dw, dh (stride 4)
//        var dx = [Float](repeating: 0, count: count)
//        var dy = [Float](repeating: 0, count: count)
//        var dw = [Float](repeating: 0, count: count)
//        var dh = [Float](repeating: 0, count: count)
//        vDSP_vindex(array, boxIndices[0], 1, &dx, 1, vDSP_Length(count))
//        vDSP_vindex(array, boxIndices[1], 1, &dy, 1, vDSP_Length(count))
//        vDSP_vindex(array, boxIndices[2], 1, &dw, 1, vDSP_Length(count))
//        vDSP_vindex(array, boxIndices[3], 1, &dh, 1, vDSP_Length(count))
//        
//        let cx = priors[0]
//        let cy = priors[1]
//        let pw = priors[2]
//        let ph = priors[3]
//        
//        let variance0: Float = 0.1
//        let variance1: Float = 0.2
//        
//        var adjX = [Float](repeating: 0, count: count)
//        var adjY = [Float](repeating: 0, count: count)
//        var expW = [Float](repeating: 0, count: count)
//        var expH = [Float](repeating: 0, count: count)
//        
//        // Compute adjusted centers
//        vDSP.multiply(dx, pw, result: &adjX)
//        vDSP.multiply(variance0, adjX, result: &adjX)
//        vDSP.add(cx, adjX, result: &adjX)
//        
//        vDSP.multiply(dy, ph, result: &adjY)
//        vDSP.multiply(variance0, adjY, result: &adjY)
//        vDSP.add(cy, adjY, result: &adjY)
//        
//        // Compute adjusted sizes
//        vDSP.multiply(variance1, dw, result: &expW)
//        vDSP.multiply(variance1, dh, result: &expH)
//        vvexpf(&expW, expW, [Int32(count)])
//        vvexpf(&expH, expH, [Int32(count)])
//        vDSP.multiply(pw, expW, result: &expW)
//        vDSP.multiply(ph, expH, result: &expH)
//        
//        // Compute (x1, y1), (x2, y2)
//        var halfW = [Float](repeating: 0, count: count)
//        var halfH = [Float](repeating: 0, count: count)
//        var divisor: Float = 2
//        vDSP_vsdiv(expW, 1, &divisor, &halfW, 1, vDSP_Length(count))
//        vDSP_vsdiv(expH, 1, &divisor, &halfH, 1, vDSP_Length(count))
//        
//        var x1 = [Float](repeating: 0, count: count)
//        var y1 = [Float](repeating: 0, count: count)
//        var x2 = [Float](repeating: 0, count: count)
//        var y2 = [Float](repeating: 0, count: count)
//        
//        vDSP.subtract(adjX, halfW, result: &x1)
//        vDSP.subtract(adjY, halfH, result: &y1)
//        vDSP.add(adjX, halfW, result: &x2)
//        vDSP.add(adjY, halfH, result: &y2)
//        
//        // Build CGRects efficiently
//        var rects = [CGRect](repeating: .zero, count: count)
//        for i in 0..<count {
//            rects[i] = CGRect(
//                x: CGFloat(x1[i]),
//                y: CGFloat(y1[i]),
//                width: CGFloat(x2[i] - x1[i]),
//                height: CGFloat(y2[i] - y1[i])
//            )
//        }
//        return rects
//    }
    
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
    
    private func extractValuesFromMLMultiArray(_ array: MLMultiArray) -> [Float] {
        let shape = array.shape.map { $0.intValue }
        let strides = array.strides.map { $0.intValue }
        precondition(shape.count == 3, "Expecting 3D MLMultiArray")
        precondition(strides[2] == 1, "Innermost stride must be contiguous")
        if #available(iOS 15.4, *) {
            return array.withUnsafeBufferPointer(ofType: Float.self) { ptr in
                var result = [Float](repeating: 0, count: shape[1] * shape[2])
                
                result.withUnsafeMutableBufferPointer { destPtr in
                    for j in 0..<shape[1] {
                        let srcIndex = j * strides[1]
                        let destIndex = j * shape[2]
                        destPtr.baseAddress!.advanced(by: destIndex)
                            .assign(from: ptr.baseAddress!.advanced(by: srcIndex), count: shape[2])
                    }
                }
                return result
            }
        } else {
            let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
            var result = [Float](repeating: 0, count: shape[1] * shape[2])
            result.withUnsafeMutableBufferPointer { destPtr in
                for j in 0..<shape[1] {
                    let srcIndex = j * strides[1]
                    let destIndex = j * shape[2]
                    destPtr.baseAddress!.advanced(by: destIndex)
                        .assign(from: ptr.advanced(by: srcIndex), count: shape[2])
                }
            }
            return result
        }
    }
    
//    private func decodeLandmarksFromArray(_ array: [Float], at indices: [Int]) -> [[CGPoint]] {
//        let count = priors[0].count
//        let variance: Float = 0.1
//        var landmarks: [[Float]] = [[Float]](repeating: [Float](repeating: .zero, count: count), count: 10)
//        for i in 0..<5 {
//            // X component
//            let dx = array[2 * i]
//            var tempW = [Float](repeating: 0, count: count)
//            var outX = [Float](repeating: 0, count: count)
//            
//            vDSP.multiply(variance, priors[2], result: &tempW)
//            vDSP.multiply(dx, tempW, result: &tempW)
//            vDSP.add(priors[0], tempW, result: &outX)
//            landmarks[2 * i] = outX
//            
//            // Y component
//            let dy = array[2 * i + 1]
//            var tempH = [Float](repeating: 0, count: count)
//            var outY = [Float](repeating: 0, count: count)
//            
//            vDSP.multiply(variance, priors[3], result: &tempH)
//            vDSP.multiply(dy, tempH, result: &tempH)
//            vDSP.add(priors[1], tempH, result: &outY)
//            landmarks[2 * i + 1] = outY
//        }
//        var pts: [[CGPoint]] = [[CGPoint]](repeating: [CGPoint](repeating: .zero, count: 5), count: count)
//        for i in 0..<count {
//            pts[i] = (0..<5).map { (j: Int) in
//                CGPoint(x: landmarks[j * 2][i], y: landmarks[j * 2 + 1][i])
//            }
//        }
//        return pts
//    }
    
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
