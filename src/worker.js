/* eslint-disable no-restricted-globals */
// In the worker, 'self' refers to the global scope.
self.onmessage = function(e) {
  const {
    type,
    data, // ArrayBuffer for the image data (Uint8ClampedArray)
    width,
    height,
    originalBuffer,
    blurredBuffer,
    adjustedPolygon,
    polygonB,
    numPixelsSimulatedBlur,
    totalBlurBlend
  } = e.data;
  
  if (type === 'processBlur') {
    // Reconstruct the typed arrays from the transferred buffers.
    let imageDataArray = new Uint8ClampedArray(data);
    const originalData = new Uint8ClampedArray(originalBuffer);
    const blurredData = new Uint8ClampedArray(blurredBuffer);
    
    // Helper: compute distance from point p to line segment vw.
    function distanceToSegment(p, v, w) {
      const l2 = (w.x - v.x) ** 2 + (w.y - v.y) ** 2;
      if (l2 === 0) return Math.hypot(p.x - v.x, p.y - v.y);
      let t = ((p.x - v.x) * (w.x - v.x) + (p.y - v.y) * (w.y - v.y)) / l2;
      t = Math.max(0, Math.min(1, t));
      const projection = {
        x: v.x + t * (w.x - v.x),
        y: v.y + t * (w.y - v.y)
      };
      return Math.hypot(p.x - projection.x, p.y - projection.y);
    }
    
    // Helper: standard point-in-polygon test; if not inside, check if within a buffer distance from any edge.
    function pointInPolygonWithBuffer(x, y, polygon, buffer = 2) {
      let inside = false;
      for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        const xi = polygon[i].x, yi = polygon[i].y;
        const xj = polygon[j].x, yj = polygon[j].y;
        if ((yi > y) !== (yj > y) &&
            (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
          inside = !inside;
        }
      }
      if (inside) return true;
      for (let i = 0; i < polygon.length; i++) {
        const v = polygon[i];
        const w = polygon[(i + 1) % polygon.length];
        const d = distanceToSegment({ x, y }, v, w);
        if (d <= buffer) return true;
      }
      return false;
    }
    
    // The heavy nested loop doing per-pixel blending.
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (
          pointInPolygonWithBuffer(x, y, adjustedPolygon, numPixelsSimulatedBlur * 3) ||
          pointInPolygonWithBuffer(x, y, polygonB, numPixelsSimulatedBlur * 3)
        ) {
          const p = { x, y };
          let minDistance = Infinity;
          // Check distance to each edge in adjustedPolygon.
          for (let i = 0; i < adjustedPolygon.length; i++) {
            const v = adjustedPolygon[i];
            const w = adjustedPolygon[(i + 1) % adjustedPolygon.length];
            const d = distanceToSegment(p, v, w);
            if (d < minDistance) minDistance = d;
          }
          // Check distance to each edge in polygonB.
          for (let i = 0; i < polygonB.length; i++) {
            const v = polygonB[i];
            const w = polygonB[(i + 1) % polygonB.length];
            const d = distanceToSegment(p, v, w);
            if (d < minDistance) minDistance = d;
          }
          const sigma = numPixelsSimulatedBlur;
          const weight = Math.exp(- (minDistance * minDistance) / (2 * sigma * sigma)) * totalBlurBlend;
          const idx = (y * width + x) * 4;
          imageDataArray[idx]     = weight * blurredData[idx]     + (1 - weight) * originalData[idx];
          imageDataArray[idx + 1] = weight * blurredData[idx + 1] + (1 - weight) * originalData[idx + 1];
          imageDataArray[idx + 2] = weight * blurredData[idx + 2] + (1 - weight) * originalData[idx + 2];
          imageDataArray[idx + 3] = weight * blurredData[idx + 3] + (1 - weight) * originalData[idx + 3];
        }
      }
    }
    
    // Send back the modified image data buffer.
    postMessage({ modifiedBuffer: imageDataArray.buffer }, [imageDataArray.buffer]);
  }
}; 