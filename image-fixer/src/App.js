import React, { useState, useEffect, useRef } from 'react';
import * as faceapi from 'face-api.js';
//import * as bodyPix from '@tensorflow-models/body-pix';

// Check the URL for ?debug=true and set the debug flag accordingly.
const debug = new URLSearchParams(window.location.search).get('debug') === 'true';
console.log("DEBUG flag is", debug);
const fillSolidColor = new URLSearchParams(window.location.search).get('fill') !== 'false';
console.log("fillSolidColor flag is", fillSolidColor);

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
  const [defaultFacesLoaded, setDefaultFacesLoaded] = useState(false);
  const [faceTargets, setFaceTargets] = useState([]); // Each element is { descriptor, image }
  const [faceScale, setFaceScale] = useState(0.9);  // New state for final face scaling (default 0.9)
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

  // NEW: Automatically load default face targets from the public/faces folder.
  useEffect(() => {
    if (modelsLoaded) {
      async function loadDefaultFaceTargets() {
        // List of default face target filenames in public/faces.
        const defaultFiles = ['elon-musk.jpg']; // <-- Update this list as needed.
        const newTargets = [];
        for (let i = 0; i < defaultFiles.length; i++) {
          const url = process.env.PUBLIC_URL + '/faces/' + defaultFiles[i];
          const img = new Image();
          img.src = url;
          // Wait for the image to load.
          await new Promise(resolve => { img.onload = resolve; });
          // Run detection on this headshot.
          const detection = await faceapi
            .detectSingleFace(img)
            .withFaceLandmarks()
            .withFaceDescriptor();
          
          if (detection) {
            newTargets.push({ descriptor: detection.descriptor, image: img });
            console.log(`Loaded default face ${defaultFiles[i]}: descriptor computed`);
          } else {
            console.warn(`No face detected in ${defaultFiles[i]}`);
          }
        }
        // Append the default face targets to any already loaded.
        setFaceTargets(prev => [...prev, ...newTargets]);
        // Mark default faces as loaded
        setDefaultFacesLoaded(true);
      }
      loadDefaultFaceTargets();
    }
  }, [modelsLoaded]);

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
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext('2d');
    // Clear the canvas so that no image is shown until the "Generate Modified Target Photo" button is clicked.
    ctx.clearRect(0, 0, img.width, img.height);
    
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
      .detectAllFaces(img)
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
        if (distance < 0.6) {
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
        
        // Compute the centroid of the face polygon.
        let sumX = 0, sumY = 0;
        facePolygon.forEach(pt => { sumX += pt.x; sumY += pt.y; });
        const centroid = { x: sumX / facePolygon.length, y: sumY / facePolygon.length };

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
        //     }
        //   } catch (error) {
        //     console.warn("Error computing head boundary polygon:", error);
        //   }
        // }
    
        // Recalculate centroid for the limited polygon.
        let limitedSumX = 0, limitedSumY = 0;
        limitedPolygon.forEach(pt => { limitedSumX += pt.x; limitedSumY += pt.y; });
        const limitedCentroid = { x: limitedSumX / limitedPolygon.length, y: limitedSumY / limitedPolygon.length };
        
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
        
        const yoffset = (rotMaxY - rotMinY) * 0.1; // the amount to stretch in the direction of the eyebrows
        const xscaling = 0.8; // the amount to narrow in the direction of the cheeks
        
        // Adjust the rotated polygon: shift each point upward based on its vertical location.
        const adjustedRotatedPolygon1 = rotatedPolygon.map(pt => {
          const weight = (rotMaxY - pt.y) / (rotMaxY - rotMinY);
          return { x: pt.x * xscaling, y: pt.y - yoffset * weight };
        });

        const yoffset2 = (rotMaxY - rotMinY) * 0.05; // the amount to shrink in the direction of the 
        
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
  
          // Compute the centroid of the polygon.
          let polyCentroid = { x: 0, y: 0 };
          translatedPolygon.forEach(pt => {
            polyCentroid.x += pt.x;
            polyCentroid.y += pt.y;
          });
          polyCentroid.x /= translatedPolygon.length;
          polyCentroid.y /= translatedPolygon.length;
  
          // For each edge of the polygon, sample points and draw a radial line from the point to the centroid.
          translatedPolygon.forEach((start, i) => {
            const end = translatedPolygon[(i + 1) % translatedPolygon.length];
            const dx = end.x - start.x;
            const dy = end.y - start.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            const steps = Math.ceil(dist);
            for (let j = 0; j <= steps; j++) {
              // Compute the pixel coordinate and clamp to valid boundaries.
              const sampleX = Math.round(start.x + j * dx / steps);
              const sampleY = Math.round(start.y + j * dy / steps);
              const x = Math.min(Math.max(sampleX, 0), imageData.width - 1);
              const y = Math.min(Math.max(sampleY, 0), imageData.height - 1);
              const index = (y * imageData.width + x) * 4;
              const r = imageData.data[index];
              const g = imageData.data[index + 1];
              const b = imageData.data[index + 2];
              const a = imageData.data[index + 3] / 255;
              compCtx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${a})`;
              compCtx.lineWidth = 1;
              compCtx.beginPath();
              compCtx.moveTo(x, y);
              compCtx.lineTo(polyCentroid.x, polyCentroid.y);
              compCtx.stroke();
            }
          });

          // --- Apply light Gaussian smoothing over the polygon area ---
          {
            // Create an offscreen canvas for the blur effect.
            const blurCanvas = document.createElement('canvas');
            blurCanvas.width = adjWidth;
            blurCanvas.height = adjHeight;
            const blurCtx = blurCanvas.getContext('2d');

            if (typeof blurCtx.filter !== 'undefined') {
              // Built-in blur filter supported – use it (desktop-friendly)
              blurCtx.filter = 'blur(4px)';
              blurCtx.save();
              blurCtx.beginPath();
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
     
        scaledFaceCtx.drawImage(faceCanvas, 0, 0, adjWidth, adjHeight, offsetX, offsetY, scaledWidth, scaledHeight);
     
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

  return (
    <div style={{ padding: '1rem', fontFamily: 'sans-serif' }}>
      <h1>Image Fixer App</h1>
      { !modelsLoaded ? (
        <p>Loading face models…</p>
      ) : !defaultFacesLoaded ? (
        <p>Loading default faces…</p>
      ) : (
        <>
          <section style={{ marginBottom: '1rem' }}>
            <h2>1. OPTIONAL: Provide individuals (headshot photos)</h2>
            <p>
              A few default faces are baked in as well.
            </p>
            <input
              type="file"
              accept="image/*"
              multiple
              onChange={handleFaceTargetsChange}
            />
          </section>

          <section style={{ marginBottom: '1rem' }}>
            <h2>2. Upload Target Photo</h2>
            <input
              type="file"
              accept="image/*"
              onChange={handleTargetPhotoChange}
            />
            <p>
              The app will detect faces and, if a face matches one of your targets,
              correct photographic errors by scaling the face by the selected percentage.
            </p>
          </section>

          <section style={{ marginBottom: '1rem' }}>
            <h2>3. Final Face Scale</h2>
            <label>
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
            <h2>4. Generate Modified Target Photo</h2>
            <button onClick={handleGenerateModifiedPhoto} style={{ marginTop: '0.5rem' }}>
              Generate Modified Target Photo
            </button>
          </section>

          <section>
            <h2>Modified Target Photo</h2>
            <canvas ref={canvasRef} style={{ maxWidth: '100%', border: '1px solid #ccc' }} />
          </section>

        </>
      )}
    </div>
  );
}

export default App;