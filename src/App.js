import React, { useState, useEffect, useRef } from 'react';
import * as faceapi from 'face-api.js';
//import * as bodyPix from '@tensorflow-models/body-pix';
import './App.css';

// Check the URL for ?debug=true and set the debug flag accordingly.
const debug = new URLSearchParams(window.location.search).get('debug') === 'true';
console.log("DEBUG flag is", debug);
const fillSolidColor = new URLSearchParams(window.location.search).get('fill') !== 'false';
console.log("fillSolidColor flag is", fillSolidColor);
const doBlur = new URLSearchParams(window.location.search).get('blur') !== 'false';
console.log("doBlur flag is", doBlur);

const eyebrowStretchAmount = 0.1;  // Amount to stretch in direction of eyebrows (was hardcoded as 0.1)
const cheekStretchAmount = 0.85;   // Amount to narrow in direction of cheeks (was hardcoded as 0.85)
const chinShrinkAmount = 0.08;     // Amount to shrink the chin (was hardcoded as 0.08)

// threshold for detecting whether something is a face or not in the image
// (0=most permissive, 1=never detects a face)
const isFaceDetection = 0.4;
// threshold for determining if a detected face matches one of our targets
// (0=never matches, 1=always matches)
const faceMatchThreshold = 0.55;

const faceLeftRightSensitivity = 3;

// NEW: Helper function to draw polygons for debugging purposes.
function drawPolygon(ctx, polygon, strokeStyle = 'red') {
  if (!polygon || polygon.length === 0) return;
  ctx.save();
  ctx.strokeStyle = strokeStyle;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(polygon[0].x, polygon[0].y);
  polygon.forEach((pt, i) => {
    if (i > 0) ctx.lineTo(pt.x, pt.y);
  });
  ctx.closePath();
  ctx.stroke();
  ctx.restore();
}

// Helper function to compute convex hull using Andrew's monotone chain algorithm.
function computeConvexHull(points) {
  if (points.length <= 1) return points.slice();
  // Clone and sort points by x, then by y.
  const sorted = points.slice().sort((a, b) => a.x === b.x ? a.y - b.y : a.x - b.x);
  const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const lower = [];
  for (const p of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }
  const upper = [];
  for (let i = sorted.length - 1; i >= 0; i--) {
    const p = sorted[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }
  // Remove the last point of each list (it's the same as the first point of the other list)
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

// Helper function to compute the average color along the boundary of a polygon.
function computeAverageBoundaryColor(imageData, polygon) {
  let totalR = 0, totalG = 0, totalB = 0, count = 0;
  for (let i = 0; i < polygon.length; i++) {
    const start = polygon[i];
    const end = polygon[(i + 1) % polygon.length];
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    const steps = Math.ceil(dist);
    for (let j = 0; j <= steps; j++) {
      const t = j / steps;
      const x = Math.floor(start.x + t * dx);
      const y = Math.floor(start.y + t * dy);
      if (x >= 0 && x < imageData.width && y >= 0 && y < imageData.height) {
        const index = (y * imageData.width + x) * 4;
        totalR += imageData.data[index];
        totalG += imageData.data[index + 1];
        totalB += imageData.data[index + 2];
        count++;
      }
    }
  }
  if (count === 0) return { r: 0, g: 0, b: 0 };
  return { r: Math.round(totalR / count), g: Math.round(totalG / count), b: Math.round(totalB / count) };
}

// NEW: Helper function to compute the centroid of an array of points.
function computeCentroid(points, faceLeftRightLookAngle = 0) {
  if (points.length === 0) return { x: 0, y: 0 };
  let sumX = 0, sumY = 0;
  points.forEach(pt => { sumX += pt.x; sumY += pt.y; });
  return { x: sumX / points.length, y: sumY / points.length };
}

// New helper function to compute the horizontal offset based on face direction.
// The function expects:
// - angle: a number between -1 (face fully left) and 1 (face fully right)
// - unshrunkWidth: the width of the original (unshrunk) face region
// - faceScale: the scale factor applied to the face (e.g., 0.9)
// 
// Calculation details:
// The total horizontal gap between the original and the scaled face is:
//   totalGap = unshrunkWidth * (1 - faceScale)
// By default, the scaled face is centered, meaning its left margin would be totalGap/2.
// To adjust the positioning based on face direction, we compute a ratio using tanh:
//   ratio = ((tanh(angle) / tanh(1)) + 1) / 2
// Then, the additional offset is computed as:
//   offset = totalGap * ratio - (totalGap/2)
// This means:
// - angle = -1: ratio becomes 0, offset = -totalGap/2 (shift left so the left edge touches)
// - angle = 0: ratio becomes 0.5, offset = 0 (centered)
// - angle = 1: ratio becomes 1, offset = totalGap/2 (shift right so the right edge touches)
function computeHorizontalOffset(angle, unshrunkWidth, faceScale) {
  const totalGap = unshrunkWidth * (1 - faceScale);
  const ratio = (Math.tanh(angle * faceLeftRightSensitivity) / Math.tanh(faceLeftRightSensitivity) + 1) / 2;
  return totalGap * ratio - totalGap / 2;
}

// Helper function to determine the gaze direction of a face based on facial landmarks.
// returns a value between -1 and 1, where -1 is looking left, 0 is straight ahead, and 1 is looking right
function getFaceDirection(landmarks) {
  // Extract eye and nose landmarks.
  const leftEye = landmarks.getLeftEye();
  const rightEye = landmarks.getRightEye();
  const nose = landmarks.getNose();

  // Compute the centers of the eyes using the computeCentroid helper.
  const leftEyeCenter = computeCentroid(leftEye);
  const rightEyeCenter = computeCentroid(rightEye);
  const eyeCenter = {
    x: (leftEyeCenter.x + rightEyeCenter.x) / 2,
    y: (leftEyeCenter.y + rightEyeCenter.y) / 2
  };

  // Determine the nose tip – choose the point with the maximum y value.
  let noseTip = nose[0];
  for (let i = 1; i < nose.length; i++) {
    if (nose[i].y > noseTip.y) {
      noseTip = nose[i];
    }
  }

  // Compute the offset between the nose tip and the eye center.
  const dx = noseTip.x - eyeCenter.x;

  // Use inter-eye distance as a reference for normalization
  const interEyeDistance = Math.hypot(rightEyeCenter.x - leftEyeCenter.x, rightEyeCenter.y - leftEyeCenter.y);

  // Normalize dx by the inter-eye distance to get a value between -1 and 1
  // Multiply by 2 since typical head rotation is about half the inter-eye distance
  let angle = (dx / interEyeDistance) * 2;

  // Clamp the value between -1 and 1
  angle = Math.max(-1, Math.min(1, angle));

  return angle;
}

// Helper function to compute the polygon for the head boundary 
// around a specified centroid in the image using the segmentation model.
async function computeHeadBoundaryPolygon(img, centroid, segmentationModel, regionSize = 200) {
  // Determine the region in the image centered on the centroid, ensuring it stays within image bounds.
  const x0 = Math.max(0, Math.min(img.width - regionSize, centroid.x - regionSize / 2));
  const y0 = Math.max(0, Math.min(img.height - regionSize, centroid.y - regionSize / 2));
  if (debug) {
    console.log("computeHeadBoundaryPolygon: region x0, y0:", x0, y0);
    console.log("computeHeadBoundaryPolygon: centroid:", centroid);
  }

  // Create a temporary canvas for the region.
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = regionSize;
  tempCanvas.height = regionSize;
  const tempCtx = tempCanvas.getContext('2d');
  tempCtx.drawImage(img, x0, y0, regionSize, regionSize, 0, 0, regionSize, regionSize);
  if (debug) {
    console.log("computeHeadBoundaryPolygon: drawn region on temporary canvas");
    // Add debug display of tempCanvas
    document.body.appendChild(tempCanvas);
    tempCanvas.style.position = 'fixed';
    tempCanvas.style.top = '10px';
    tempCanvas.style.right = '10px';
    tempCanvas.style.border = '2px solid red';
    tempCanvas.style.zIndex = '9999';
  }

  console.log("computeHeadBoundaryPolygon: segmentationModel", segmentationModel);
  // Run segmentation on this region.
  const segmentation = await segmentationModel.segmentPerson(tempCanvas, { internalResolution: 'medium' });
  if (debug) {
    console.log("computeHeadBoundaryPolygon: segmentation result:", segmentation);
    console.log("computeHeadBoundaryPolygon: segmentation.data.length =", segmentation.data.length);
  }

  // Collect person pixel coordinates.
  const personPoints = [];
  for (let i = 0; i < segmentation.data.length; i++) {
    if (segmentation.data[i] === 1) {  // bodyPix returns 1 for person pixels.
      const px = i % regionSize;
      const py = Math.floor(i / regionSize);
      personPoints.push({ x: px, y: py });
    }
  }
  if (debug) {
    console.log("computeHeadBoundaryPolygon: number of person pixels =", personPoints.length);
  }

  if (personPoints.length === 0) {
    if (debug) { console.log("computeHeadBoundaryPolygon: no person pixels found, falling back to rectangular region."); }
    // Fallback: return the rectangular region if no person detected.
    return [
      { x: x0, y: y0 },
      { x: x0 + regionSize, y: y0 },
      { x: x0 + regionSize, y: y0 + regionSize },
      { x: x0, y: y0 + regionSize }
    ];
  }

  // Compute the convex hull of the person pixels.
  const hull = computeConvexHull(personPoints);
  if (debug) {
    console.log("computeHeadBoundaryPolygon: convex hull computed:", hull);
  }

  // Map the hull coordinates back to the full image coordinate system.
  const mappedHull = hull.map(pt => ({
    x: pt.x + x0,
    y: pt.y + y0
  }));
  if (debug) {
    console.log("computeHeadBoundaryPolygon: mapped hull:", mappedHull);
  }

  return mappedHull;
}

// NEW: Helper function to compute the intersection of two line segments.
function lineIntersection(p, q, a, b) {
  const A1 = q.y - p.y;
  const B1 = p.x - q.x;
  const C1 = A1 * p.x + B1 * p.y;

  const A2 = b.y - a.y;
  const B2 = a.x - b.x;
  const C2 = A2 * a.x + B2 * a.y;

  const determinant = A1 * B2 - A2 * B1;
  if (determinant === 0) {
    return null; // lines are parallel
  } else {
    const x = (B2 * C1 - B1 * C2) / determinant;
    const y = (A1 * C2 - A2 * C1) / determinant;
    return { x, y };
  }
}

// NEW: Helper function to compute the intersection (clipping) of two convex polygons.
function polygonIntersection(subjectPolygon, clipPolygon) {
  let outputList = subjectPolygon;
  for (let i = 0; i < clipPolygon.length; i++) {
    const inputList = outputList;
    outputList = [];
    const A = clipPolygon[i];
    const B = clipPolygon[(i + 1) % clipPolygon.length];
    const edgeVec = { x: B.x - A.x, y: B.y - A.y };

    const inside = (point) => {
      return (edgeVec.x * (point.y - A.y) - edgeVec.y * (point.x - A.x)) >= 0;
    };

    for (let j = 0; j < inputList.length; j++) {
      const P = inputList[j];
      const Q = inputList[(j + 1) % inputList.length];
      const pInside = inside(P);
      const qInside = inside(Q);
      if (qInside) {
        if (!pInside) {
          const intersect = lineIntersection(P, Q, A, B);
          if (intersect) outputList.push(intersect);
        }
        outputList.push(Q);
      } else if (pInside) {
        const intersect = lineIntersection(P, Q, A, B);
        if (intersect) outputList.push(intersect);
      }
    }
  }
  return outputList;
}

function App() {
  // Use state to track if models are loaded and to store the descriptors for our target headshots.
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [faceTargets, setFaceTargets] = useState([]); // Each element is { descriptor, image }
  const [faceScale, setFaceScale] = useState(0.85);  // New state for final face scaling (default 0.9)
  
  // NEW: States for default face management:
  const [defaultFaceOptions, setDefaultFaceOptions] = useState([]);  // Array of { filename, name, rank }
  const [selectedDefaultFaceFilenames, setSelectedDefaultFaceFilenames] = useState([]); // The filenames selected from dropdown
  const [faceSearchTerm, setFaceSearchTerm] = useState("");  // For filtering the dropdown

  // NEW: States for tracking default faces loading progress.
  const [defaultFacesProgress, setDefaultFacesProgress] = useState(0);
  const [totalDefaultFaces, setTotalDefaultFaces] = useState(0);
  
  // NEW state for segmentation model
  const [segmentationModel, setSegmentationModel] = useState(null);
  // NEW: State for storing the uploaded target photo.
  const [targetImage, setTargetImage] = useState(null);
  // Use references for the canvas and (optionally) the image elements.
  const canvasRef = useRef(null);

  // Load face-api models from the public folder.
  useEffect(() => {
    const MODEL_URL = process.env.PUBLIC_URL + '/models';
    Promise.all([
      faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
      faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
      faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL)
    ])
      .then(() => {
        console.log('Face models loaded');
        setModelsLoaded(true);
      })
      .catch(err => console.error("Error loading models", err));
  }, []);

  // // NEW: Load segmentation model (e.g. BodyPix)
  // useEffect(() => {
  //   async function loadSegModel() {
  //     const model = await bodyPix.load();
  //     setSegmentationModel(model);
  //     console.log("Segmentation model loaded");
  //   }
  //   loadSegModel();
  // }, []);

  // NEW: Load default face options from face_filenames.json for the dropdown.
  useEffect(() => {
    async function loadDefaultFaceOptions() {
      try {
        const response = await fetch(process.env.PUBLIC_URL + '/face_filenames.json');
        if (!response.ok) {
          console.error("Failed to load face_filenames.json");
          return;
        }
        const jsonData = await response.json();
        // Assuming jsonData structure: { "filename1.png": { name: "Name1", rank: 50 }, ... }
        // Filter out entries where rank is not a valid number and convert the rank to a number.
        const options = Object.keys(jsonData)
          .filter(filename => !isNaN(parseFloat(jsonData[filename].rank)))
          .map(filename => ({
            filename,
            name: jsonData[filename].name,
            rank: parseFloat(jsonData[filename].rank)
          }));
        // Sort options in ascending order by rank.
        options.sort((a, b) => a.rank - b.rank);
        setDefaultFaceOptions(options);
        setTotalDefaultFaces(options.length); // Optionally update total default faces
      } catch (error) {
        console.error("Error fetching face_filenames.json:", error);
      }
    }
    loadDefaultFaceOptions();
  }, []);

  // NEW: Load default faces on demand when the user selects them using checkboxes.
  // We use a ref to track which default faces have already been loaded.
  const loadedDefaultSetRef = useRef(new Set());

  useEffect(() => {
    if (!modelsLoaded) return;

    async function fetchDefaultFaces() {
      // Remove any default face from faceTargets that is no longer selected.
      setFaceTargets(prev =>
        prev.filter(target => target.source !== 'default' || selectedDefaultFaceFilenames.includes(target.filename))
      );

      // Determine which selected filenames have not been loaded yet.
      const toLoad = selectedDefaultFaceFilenames.filter(
        filename => !loadedDefaultSetRef.current.has(filename)
      );

      if (toLoad.length === 0) return;

      for (const filename of toLoad) {
        const url = process.env.PUBLIC_URL + '/faces/' + filename;
        const img = new Image();
        img.src = url;
        await new Promise(resolve => {
          img.onload = resolve;
        });

        const detection = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor();

        if (detection) {
          setFaceTargets(prev => [
            ...prev,
            {
              descriptor: detection.descriptor,
              image: img,
              filename,
              source: 'default'
            }
          ]);
          loadedDefaultSetRef.current.add(filename);
          console.log(`Loaded default face ${filename}`);
          setDefaultFacesProgress(prev => prev + 1);
        } else {
          console.warn(`No face detected in ${filename}`);
        }
      }
    }

    fetchDefaultFaces();
  }, [selectedDefaultFaceFilenames, modelsLoaded]);

  // UPDATED: Append uploaded face targets to existing ones.
  const handleFaceTargetsChange = async (event) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;
    const targets = [];

    // Process each uploaded image
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const img = new Image();
      img.src = URL.createObjectURL(file);
      // Wait for the image to load.
      await new Promise(resolve => { img.onload = resolve; });
      
      // Run detection on this headshot.
      const detection = await faceapi
        .detectSingleFace(img)
        .withFaceLandmarks()
        .withFaceDescriptor();
      
      if (detection) {
        targets.push({ descriptor: detection.descriptor, image: img });
        console.log(`Loaded uploaded target ${i}: descriptor computed`);
      } else {
        console.warn(`No face detected in ${file.name}`);
      }
    }
    // Append the new targets to the existing face targets.
    setFaceTargets(prev => [...prev, ...targets]);
  };

  // Handler for uploading the target photo where some faces might be in the face targets
  const handleTargetPhotoChange = async (event) => {
    console.log("handleTargetPhotoChange triggered");
    const file = event.target.files[0];
    if (!file) return;
    
    // Create an image element for the target photo.
    const img = new Image();
    img.src = URL.createObjectURL(file);
    await new Promise(resolve => { img.onload = resolve; });

    // Store the loaded image in state for later processing.
    setTargetImage(img);
    
    // Get the canvas ref and size it to the photo.
    const canvas = canvasRef.current;
    // Use the image's intrinsic dimensions.
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d');
    // Optionally clear previous canvas content.
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Redraw the original target photo at full resolution.
    //ctx.drawImage(img, 0, 0, img.naturalWidth, img.naturalHeight);
    
    // Removed the automatic processing of the target photo.
  };

  // NEW: Handler to generate the modified target photo using the stored targetImage.
  const handleGenerateModifiedPhoto = async () => {
    if (!targetImage) {
      console.warn("No target image available to process");
      return;
    }
    
    const img = targetImage;
    const canvas = canvasRef.current;
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext('2d');
    
    // Optionally clear previous canvas content.
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Redraw the original target photo.
    ctx.drawImage(img, 0, 0, img.width, img.height);
    
    // Run face detection (with landmarks and descriptors) on the target image.
    const detections = await faceapi
      .detectAllFaces(img, new faceapi.SsdMobilenetv1Options({ minConfidence: isFaceDetection }))
      .withFaceLandmarks()
      .withFaceDescriptors();

    console.log(`Detected ${detections.length} faces in the target photo`);

    // Process each detected face.
    for (let index = 0; index < detections.length; index++) {
      const det = detections[index];
      const descriptor = det.descriptor;
      let isTargetMatch = false;
      
      // Compare with each face target.
      for (const target of faceTargets) {
        const distance = faceapi.euclideanDistance(descriptor, target.descriptor);
        if (distance < faceMatchThreshold) {
          isTargetMatch = true;
          break;
        }
      }
      
      if (isTargetMatch) {
        console.log(`Face ${index} matches a face target – processing with polygon mask.`);
        
        // Get points from jaw outline and eyebrows.
        const jawOutline = det.landmarks.getJawOutline();
        const leftEyebrow = det.landmarks.getLeftEyeBrow();
        const rightEyebrow = det.landmarks.getRightEyeBrow();
        const allPoints = [...jawOutline, ...leftEyebrow, ...rightEyebrow];
        // Compute the convex hull of these points to get a polygon that encloses the face.
        const facePolygon = computeConvexHull(allPoints);
        
        // Compute the centroid of the face polygon using the helper function.
        const centroid = computeCentroid(facePolygon);

        // Compute the head boundary polygon using the segmentation model and clip the face polygon.
        let limitedPolygon = facePolygon;
        // if (segmentationModel) {
        //   try {
        //     console.log("Computing head boundary polygon");
        //     const headPolygon = await computeHeadBoundaryPolygon(img, centroid, segmentationModel);
        //     console.log("Head boundary polygon computed");
        //     const clippedPolygon = polygonIntersection(facePolygon, headPolygon);
        //     console.log("Clipped polygon computed");
        //     if (clippedPolygon.length > 0) {
        //       limitedPolygon = clippedPolygon;
        //     } else {
        //       console.warn("Clipping result is empty, using original face polygon.");
        //     }
        //     if (debug) {
        //       console.log('Face polygon:', facePolygon);
        //       console.log('Head polygon:', headPolygon);
        //       console.log('Clipped polygon:', clippedPolygon);

        //       // Draw debug overlays on the main canvas.
        //       // facePolygon in red, headPolygon in green, clipped polygon in blue.
        //       drawPolygon(ctx, facePolygon, 'red');
        //       drawPolygon(ctx, headPolygon, 'green');
        //       drawPolygon(ctx, clippedPolygon, 'blue');

        //       // Draw debug overlays on the main canvas.
        //       // facePolygon in red, headPolygon in green, clipped polygon in blue.
        //       drawPolygon(ctx, facePolygon, 'red');
        //       drawPolygon(ctx, headPolygon, 'green');
        //       drawPolygon(ctx, clippedPolygon, 'blue');
        //     }
        //   } catch (error) {
        //     console.warn("Error computing head boundary polygon:", error);
        //   }
        // }
    
        // Recalculate centroid for the limited polygon using the helper function.
        const limitedCentroid = computeCentroid(limitedPolygon);
        
        // Get eye positions to compute face rotation.
        const leftEye = det.landmarks.getLeftEye();
        const rightEye = det.landmarks.getRightEye();
        let leftEyeCenter = { x: 0, y: 0 }, rightEyeCenter = { x: 0, y: 0 };
        leftEye.forEach(p => { leftEyeCenter.x += p.x; leftEyeCenter.y += p.y; });
        rightEye.forEach(p => { rightEyeCenter.x += p.x; rightEyeCenter.y += p.y; });
        leftEyeCenter.x /= leftEye.length;
        leftEyeCenter.y /= leftEye.length;
        rightEyeCenter.x /= rightEye.length;
        rightEyeCenter.y /= rightEye.length;

        // Get face direction using the helper function
        const faceDirection = getFaceDirection(det.landmarks);
        console.log("Face direction:", faceDirection);
        
        // Compute the face rotation angle (in radians) from the eye centers.
        const angle = Math.atan2(rightEyeCenter.y - leftEyeCenter.y, rightEyeCenter.x - leftEyeCenter.x);
        
        // Rotate the limited polygon by -angle around the limitedCentroid to align it vertically.
        const rotatedPolygon = limitedPolygon.map(pt => {
          const dx = pt.x - limitedCentroid.x;
          const dy = pt.y - limitedCentroid.y;
          return {
            x: dx * Math.cos(angle) + dy * Math.sin(angle),
            y: -dx * Math.sin(angle) + dy * Math.cos(angle)
          };
        });
        
        // Compute the vertical bounds of the rotated polygon.
        let rotMinY = Infinity, rotMaxY = -Infinity;
        rotatedPolygon.forEach(pt => {
          rotMinY = Math.min(rotMinY, pt.y);
          rotMaxY = Math.max(rotMaxY, pt.y);
        });
        
        const yoffset = (rotMaxY - rotMinY) * eyebrowStretchAmount; // the amount to stretch in the direction of the eyebrows
        const xscaling = cheekStretchAmount; // the amount to narrow in the direction of the cheeks
        
        // Adjust the rotated polygon: shift each point upward based on its vertical location.
        const adjustedRotatedPolygon1 = rotatedPolygon.map(pt => {
          const weight = (rotMaxY - pt.y) / (rotMaxY - rotMinY);
          return { x: pt.x * xscaling, y: pt.y - yoffset * weight };
        });

        const yoffset2 = (rotMaxY - rotMinY) * chinShrinkAmount; // the amount to shrink the chin
        
        // Adjust the rotated polygon: shift each point upward based on its vertical location.
        const adjustedRotatedPolygon = adjustedRotatedPolygon1.map(pt => {
          const weight = (pt.y - rotMinY) / (rotMaxY - rotMinY);
          return { x: pt.x, y: pt.y - yoffset2 * weight };
        });
        
        // Rotate the adjusted polygon back by +angle (inverse rotation) around the limitedCentroid.
        const adjustedPolygon = adjustedRotatedPolygon.map(pt => {
          return {
            x: limitedCentroid.x + pt.x * Math.cos(angle) - pt.y * Math.sin(angle),
            y: limitedCentroid.y + pt.x * Math.sin(angle) + pt.y * Math.cos(angle)
          };
        });
        
        // Compute the bounding box of the adjusted polygon.
        let adjMinX = Infinity, adjMinY = Infinity, adjMaxX = -Infinity, adjMaxY = -Infinity;
        adjustedPolygon.forEach(pt => {
          adjMinX = Math.min(adjMinX, pt.x);
          adjMinY = Math.min(adjMinY, pt.y);
          adjMaxX = Math.max(adjMaxX, pt.x);
          adjMaxY = Math.max(adjMaxY, pt.y);
        });
        const adjWidth = adjMaxX - adjMinX;
        const adjHeight = adjMaxY - adjMinY;
        
        // Translate the adjusted polygon to the offscreen canvas coordinate system.
        const translatedPolygon = adjustedPolygon.map(pt => ({
          x: pt.x - adjMinX,
          y: pt.y - adjMinY
        }));
        
        let compCanvas, compCtx;
        if (fillSolidColor) {
          // Create a temporary canvas to extract the face region.
          const tempCanvas = document.createElement('canvas');
          tempCanvas.width = adjWidth;
          tempCanvas.height = adjHeight;
          const tempCtx = tempCanvas.getContext('2d');
          // Disable image smoothing on the temp canvas to get raw pixel values.
          tempCtx.imageSmoothingEnabled = false;
          tempCtx.drawImage(img, adjMinX, adjMinY, adjWidth, adjHeight, 0, 0, adjWidth, adjHeight);
          const imageData = tempCtx.getImageData(0, 0, adjWidth, adjHeight);
  
          // Create an offscreen composite canvas for the radial effect.
          compCanvas = document.createElement('canvas');
          compCanvas.width = adjWidth;
          compCanvas.height = adjHeight;
          compCtx = compCanvas.getContext('2d');
          // Disable smoothing when drawing the lines.
          compCtx.imageSmoothingEnabled = false;
          // Use settings that avoid anti-aliasing as much as possible.
          compCtx.lineCap = 'butt';
          compCtx.lineJoin = 'miter';
  
          // Compute the centroid of the polygon using the helper function.
          const polyCentroid = computeCentroid(translatedPolygon, faceDirection);
          // Compute horizontal offset based on face direction and adjust the centroid.
          const horizontalOffset = computeHorizontalOffset(faceDirection, adjWidth, faceScale);
          const adjustedPolyCentroid = { x: polyCentroid.x + horizontalOffset, y: polyCentroid.y };
  
          // For each edge of the polygon, sample points and draw a radial line from the point to the centroid.
          translatedPolygon.forEach((start, i) => {
            const end = translatedPolygon[(i + 1) % translatedPolygon.length];
            const dx = end.x - start.x;
            const dy = end.y - start.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            const steps = Math.ceil(dist);
            const numSurroundingPixels = 4;
            for (let j = 0; j <= steps; j++) {
              // Compute the pixel coordinate and clamp to valid boundaries.
              const sampleX = Math.round(start.x + j * dx / steps);
              const sampleY = Math.round(start.y + j * dy / steps);
              const x = Math.min(Math.max(sampleX, 0), imageData.width - 1);
              const y = Math.min(Math.max(sampleY, 0), imageData.height - 1);
              // Average the color of the pixel at (x, y) with its surrounding neighbors.
              let sumR = 0, sumG = 0, sumB = 0, sumA = 0, count = 0;
              for (let offsetY = -numSurroundingPixels; offsetY <= numSurroundingPixels; offsetY++) {
                for (let offsetX = -numSurroundingPixels; offsetX <= numSurroundingPixels; offsetX++) {
                  // Clamp the neighbor coordinates to the image boundaries.
                  const sampleX = Math.min(Math.max(x + offsetX, 0), imageData.width - 1);
                  const sampleY = Math.min(Math.max(y + offsetY, 0), imageData.height - 1);
                  const idx = (sampleY * imageData.width + sampleX) * 4;
                  sumR += imageData.data[idx];
                  sumG += imageData.data[idx + 1];
                  sumB += imageData.data[idx + 2];
                  sumA += imageData.data[idx + 3];
                  count++;
                }
              }
              const r = sumR / count;
              const g = sumG / count;
              const b = sumB / count;
              const a = (sumA / count) / 255;
              compCtx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${a})`;
              compCtx.lineWidth = 1;
              compCtx.beginPath();
              compCtx.moveTo(x, y);
              // Draw radial line to the adjusted centroid.
              compCtx.lineTo(adjustedPolyCentroid.x, adjustedPolyCentroid.y);
              compCtx.stroke();
            }
          });

          // --- Apply light Gaussian smoothing over the polygon area ---
          if (doBlur) {
            // Create an offscreen canvas for the blur effect.
            const blurCanvas = document.createElement('canvas');
            blurCanvas.width = adjWidth;
            blurCanvas.height = adjHeight;
            const blurCtx = blurCanvas.getContext('2d');

            if (typeof blurCtx.filter !== 'undefined') {
              // Built-in blur filter supported – use it (desktop-friendly)
              blurCtx.filter = 'blur(4px)';
              blurCtx.save();
              translatedPolygon.forEach((pt, i) => {
                if (i === 0) blurCtx.moveTo(pt.x, pt.y);
                else blurCtx.lineTo(pt.x, pt.y);
              });
              blurCtx.closePath();
              blurCtx.clip();
              blurCtx.drawImage(compCanvas, 0, 0);
              blurCtx.restore();

              // Overlay the blurred result back onto the composite canvas.
              compCtx.save();
              compCtx.beginPath();
              translatedPolygon.forEach((pt, i) => {
                if (i === 0) compCtx.moveTo(pt.x, pt.y);
                else compCtx.lineTo(pt.x, pt.y);
              });
              compCtx.closePath();
              compCtx.clip();
              compCtx.drawImage(blurCanvas, 0, 0);
              compCtx.restore();
            } else {
              // Fallback for mobile browsers lacking support for ctx.filter.
              // Simulate a blur by drawing the image multiple times with slight offsets.
              const tempCanvas = document.createElement('canvas');
              tempCanvas.width = adjWidth;
              tempCanvas.height = adjHeight;
              const tempCtx = tempCanvas.getContext('2d');
              tempCtx.drawImage(compCanvas, 0, 0);

              // Clear the area where the blur will be applied.
              compCtx.clearRect(0, 0, adjWidth, adjHeight);

              // Clip to the polygon area on compCtx.
              compCtx.save();
              compCtx.beginPath();
              translatedPolygon.forEach((pt, i) => {
                if (i === 0) compCtx.moveTo(pt.x, pt.y);
                else compCtx.lineTo(pt.x, pt.y);
              });
              compCtx.closePath();
              compCtx.clip();

              // Loop over small offsets to simulate a blur effect.
              const offsets = [-2, -1, 0, 1, 2];
              offsets.forEach(dx => {
                offsets.forEach(dy => {
                  compCtx.globalAlpha = 0.1; // lowered opacity for smooth accumulation
                  compCtx.drawImage(tempCanvas, dx, dy);
                });
              });
              compCtx.globalAlpha = 1.0;
              compCtx.restore();
            }
          }
        }
        
        // Step 2. Scale the masked face from faceCanvas by faceScale into a new canvas (scaledFaceCanvas).
        const faceCanvas = document.createElement('canvas');
        faceCanvas.width = adjWidth;
        faceCanvas.height = adjHeight;
        const faceCtx = faceCanvas.getContext('2d');
     
        faceCtx.save();
        faceCtx.beginPath();
        translatedPolygon.forEach((pt, i) => {
          if (i === 0) {
            faceCtx.moveTo(pt.x, pt.y);
          } else {
            faceCtx.lineTo(pt.x, pt.y);
          }
        });
        faceCtx.closePath();
        faceCtx.clip();
     
        // Draw the full face region into faceCanvas.
        faceCtx.drawImage(img, adjMinX, adjMinY, adjWidth, adjHeight, 0, 0, adjWidth, adjHeight);
        faceCtx.restore();
     
        // Step 2. Scale the masked face from faceCanvas by faceScale into a new canvas (scaledFaceCanvas).
        const scaledFaceCanvas = document.createElement('canvas');
        scaledFaceCanvas.width = adjWidth;
        scaledFaceCanvas.height = adjHeight;
        const scaledFaceCtx = scaledFaceCanvas.getContext('2d');
     
        const scaledWidth = adjWidth * faceScale;
        const scaledHeight = adjHeight * faceScale;
        const offsetX = (adjWidth - scaledWidth) / 2;
        const offsetY = (adjHeight - scaledHeight) / 2;
        // Compute additional horizontal offset based on face direction.
        const additionalOffset = computeHorizontalOffset(faceDirection, adjWidth, faceScale);
        const newOffsetX = offsetX + additionalOffset;
     
        scaledFaceCtx.drawImage(faceCanvas, 0, 0, adjWidth, adjHeight, newOffsetX, offsetY, scaledWidth, scaledHeight);
     
        if (fillSolidColor) {
          // Step 3. On the composite canvas, overlay the scaled, masked face on top of the average-color background.
          compCtx.drawImage(scaledFaceCanvas, 0, 0);
          // Finally, draw the composite canvas onto the main canvas.
          ctx.drawImage(compCanvas, 0, 0, adjWidth, adjHeight, adjMinX, adjMinY, adjWidth, adjHeight);
        } else {
          // If not filling with a solid color, simply draw the scaled face directly on top of the original.
          ctx.drawImage(scaledFaceCanvas, 0, 0, adjWidth, adjHeight, adjMinX, adjMinY, adjWidth, adjHeight);
        }
      }
    }
  };

  useEffect(() => {
    document.title = "Billionaire Face Shrinker";
  }, []);

  return (
    <div className="app-container">
      <h1 className="logo">Billionaire Face $hrinker</h1>
      <p className="subtitle">because they are deeply unserious people, and should be shrunk.</p>
      <br></br>
      { !modelsLoaded ? (
        <p>Loading face models…</p>
      ) : (
        <>
          <section style={{ marginBottom: '1rem' }}>
            <h2>1. Upload Photos to Faceshrink</h2>
            <input
              type="file"
              accept="image/*"
              onChange={handleTargetPhotoChange}
            />
          </section>

          <section style={{ marginBottom: '1rem' }}>
            <h2>2. Select Billionaires to Faceshrink</h2>
            <input
              type="text"
              placeholder="Search faces…"
              value={faceSearchTerm}
              onChange={(e) => setFaceSearchTerm(e.target.value)}
              style={{ marginBottom: '0.5rem', display: 'block' }}
            />
            <div style={{ maxHeight: '200px', overflowY: 'auto', border: '1px solid #ddd', padding: '0.5rem', marginBottom: '1rem' }}>
              {defaultFaceOptions
                .filter(option =>
                  option.name.toLowerCase().includes(faceSearchTerm.toLowerCase())
                )
                .map(option => (
                  <label key={option.filename} style={{ display: 'block', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      value={option.filename}
                      checked={selectedDefaultFaceFilenames.includes(option.filename)}
                      onChange={() => {
                        // Only add the filename if not already selected.
                        if (!selectedDefaultFaceFilenames.includes(option.filename)) {
                          setSelectedDefaultFaceFilenames([...selectedDefaultFaceFilenames, option.filename]);
                        }
                      }}
                      disabled={selectedDefaultFaceFilenames.includes(option.filename)}
                      style={{ marginRight: '0.5rem' }}
                    />
                    {option.name} (Rank: {option.rank})
                  </label>
                ))}
            </div>
            <br></br>
            <h3 style={{ marginTop: '0rem' }}>(OPTIONAL: Provide additional individuals to faceshrink (headshot photos))</h3>
            <input
              type="file"
              accept="image/*"
              multiple
              onChange={handleFaceTargetsChange}
            />
          </section>

          <section style={{ marginBottom: '1rem' }}>
            <h2>3. Final Face Scale</h2>
            <label className="face-scale-label">
              Scale: {faceScale}
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.01"
                value={faceScale}
                onChange={e => setFaceScale(parseFloat(e.target.value))}
              />
            </label>
          </section>


          <section style={{ marginBottom: '1rem' }}>
            <h2>4. Generate!</h2>
            <button className="big-button" onClick={handleGenerateModifiedPhoto} style={{ marginTop: '0.5rem' }}>
              Generate
            </button>
          </section>

          <section>
            <canvas 
              ref={canvasRef} 
              className="face-canvas"
            />
          </section>

          <section style={{ marginBottom: '1rem' }}>
            <h2>Contact</h2>
            <p>
              Bluesky:&nbsp;
              <a href="https://bsky.app/profile/seelematic.bsky.social" target="_blank" rel="noopener noreferrer">
                @seelematic.bsky.social
              </a>
            </p>
          </section>
        </>
      )}
    </div>
  );
}

export default App;