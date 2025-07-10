import XCTest
import VerIDCommonTypes
import CoreML
@testable import FaceDetectionRetinaFace

final class FaceDetectionRetinaFaceTests: XCTestCase {
    
    var faceDetection: FaceDetectionRetinaFace!
    
    override func setUpWithError() throws {
        self.faceDetection = try FaceDetectionRetinaFace()
    }
    
    func testDetectFaceInImage() async throws {
        guard let url = Bundle.module.url(forResource: "image", withExtension: "jpg", subdirectory: nil) else {
            XCTFail("Invalid image URL")
            return
        }
        let data = try Data(contentsOf: url)
        guard let cgImage = UIImage(data: data)?.cgImage, let image = Image(cgImage: cgImage) else {
            XCTFail("Failed to create image")
            return
        }
        guard let face = try await self.faceDetection.detectFacesInImage(image, limit: 1).first else {
            XCTFail("Failed to detect a face in image")
            return
        }
        guard let faceJsonURL = Bundle.module.url(forResource: "face", withExtension: "json") else {
            XCTFail()
            return
        }
        let faceJsonData = try Data(contentsOf: faceJsonURL)
        let expectedFace = try JSONDecoder().decode(Face.self, from: faceJsonData)
        XCTAssertEqual(face, expectedFace)
//        let annotatedImage = self.renderFace(face, onImage: cgImage)
//        let attachment = XCTAttachment(image: annotatedImage)
//        attachment.name = "Face"
//        attachment.lifetime = .keepAlways
//        self.add(attachment)
    }
    
    func testScalePixelBuffer() throws {
        guard let url = Bundle.module.url(forResource: "image", withExtension: "jpg", subdirectory: nil) else {
            XCTFail("Invalid image URL")
            return
        }
        let data = try Data(contentsOf: url)
        guard let cgImage = UIImage(data: data)?.cgImage, let image = Image(cgImage: cgImage) else {
            XCTFail("Failed to create image")
            return
        }
        let modelPrep = Preprocessing()
        let targetSize = CGSize(width: 640, height: 640)
        let (scaled, scale) = try modelPrep.scalePixelBuffer(image.videoBuffer, to: targetSize)
        let ciImage = CIImage(cvPixelBuffer: scaled)
        XCTAssertEqual(ciImage.extent.width, targetSize.width)
        XCTAssertEqual(ciImage.extent.height, targetSize.height)
        XCTAssertEqual(scale, targetSize.width / CGFloat(max(cgImage.width, cgImage.height)), accuracy: 0.001)
//        let att = XCTAttachment(image: UIImage(ciImage: CIImage(cvPixelBuffer: scaled)))
//        att.name = "Scaled image"
//        att.lifetime = .keepAlways
//        self.add(att)
    }
    
    private func renderFace(_ face: Face, onImage image: CGImage) -> UIImage {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1.0
        let annotatedImage = UIGraphicsImageRenderer(size: CGSize(width: image.width, height: image.height), format: format).image { context in
            UIImage(cgImage: image).draw(at: .zero)
            UIColor.green.setStroke()
            let path = UIBezierPath(rect: face.bounds)
            path.lineWidth = 10
            path.stroke()
            let labeledLandmarks: [(String,CGPoint)] = [
                ("left eye", face.leftEye),
                ("right eye", face.rightEye),
                ("nose", face.noseTip!),
                ("mouth left", face.mouthLeftCorner!),
                ("mouth right", face.mouthRightCorner!)
            ]
            UIColor.green.setFill()
            let dotRadius: CGFloat = 5
            for (label, landmark) in labeledLandmarks {
                UIBezierPath(arcCenter: landmark, radius: dotRadius, startAngle: 0, endAngle: .pi * 2, clockwise: true).fill()
                let attributes: [NSAttributedString.Key: Any] = [
                    .font: UIFont.systemFont(ofSize: 12),
                    .foregroundColor: UIColor.red
                ]
                
                let labelText = NSString(string: label)
                let textSize = labelText.size(withAttributes: attributes)
                
                let labelOrigin = CGPoint(
                    x: landmark.x - textSize.width / 2,
                    y: landmark.y - dotRadius - textSize.height - 2 // 2pt padding above the dot
                )
                
                labelText.draw(at: labelOrigin, withAttributes: attributes)
            }
        }
        return annotatedImage
    }
}
