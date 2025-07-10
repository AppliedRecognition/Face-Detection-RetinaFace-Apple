//
//  Errors.swift
//
//
//  Created by Jakub Dolejs on 09/07/2025.
//

import Foundation

public enum FaceDetectionRetinaFaceError: LocalizedError {
    case missingExpectedModelOutputs
    case imageResizingError
    
    public var errorDescription: String? {
        switch self {
        case .missingExpectedModelOutputs:
            return NSLocalizedString("Missing expected model outputs", comment: "")
        case .imageResizingError:
            return NSLocalizedString("Failed to resize input image", comment: "")
        }
    }
}
