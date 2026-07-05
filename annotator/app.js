const map = L.map('map').setView([25.485, 30.565], 16);

const esri = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    attribution: 'Tiles &copy; Esri',
    maxNativeZoom: 17,
    maxZoom: 25
});

const googleSat = L.tileLayer('http://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', {
    maxZoom: 25,
    subdomains: ['mt0', 'mt1', 'mt2', 'mt3']
});

esri.addTo(map);

L.control.layers({
    "Esri Satellite": esri,
    "Google Satellite": googleSat
}).addTo(map);

let approvedCount = 0;
let rejectedCount = 0;
let totalCount = 0;
let statusMap = new Map(); // id -> status
let manualFeatures = []; // To store manually drawn lines
let currentDoorLayerGroup = L.layerGroup().addTo(map);
let isDrawing = false;
let isSelectionMode = true; // default to true
let drawStartPoint = null;

const updateStats = () => {
    document.getElementById('total-val').innerText = totalCount;
    document.getElementById('approved-val').innerText = approvedCount;
    document.getElementById('rejected-val').innerText = rejectedCount;
};

// Load Buildings for Hover
fetch('buildings.geojson')
    .then(res => res.json())
    .then(data => {
        L.geoJSON(data, {
            style: { color: '#ffffff', weight: 1, fillOpacity: 0.1, opacity: 0.5 },
            onEachFeature: (feature, layer) => {
                const id = feature.properties.chapel_id || feature.properties.DXF_ID || feature.properties.Id || "Unknown";
                layer.bindTooltip(`Chapel ID: ${id}`, { sticky: true, className: 'custom-tooltip' });
            }
        }).addTo(map);
    });

// Load DXF Walls
fetch('walls.geojson')
    .then(res => res.json())
    .then(data => {
        L.geoJSON(data, { style: { color: '#3b82f6', weight: 2, opacity: 0.8 } }).addTo(map);
        const bounds = L.geoJSON(data).getBounds();
        map.fitBounds(bounds);
        map.setMaxBounds(bounds.pad(0.2));
        map.options.minZoom = 15;
    });

const createInteractiveDoor = (feature, layer, isManual=false) => {
    const id = feature.id || Math.random().toString(36).substr(2, 9);
    feature.id = id;
    statusMap.set(id, 'pending');
    
    const bounds = layer.getBounds();
    const center = bounds.getCenter();
    const dot = L.circleMarker(center, {
        radius: 4, fillColor: '#f59e0b', color: '#fff', weight: 1, opacity: 1, fillOpacity: 1
    }).addTo(currentDoorLayerGroup);

    layer.addTo(currentDoorLayerGroup);

    const toggleAudit = function(e) {
        if (isDrawing || !isSelectionMode) return; // don't toggle if drawing or not in selection mode
        const status = statusMap.get(id);
        if (status === 'pending' || status === 'rejected') {
            statusMap.set(id, 'approved');
            layer.setStyle({ color: '#10b981' });
            dot.setStyle({ fillColor: '#10b981' });
            if(status === 'rejected') rejectedCount--;
            if(status !== 'approved') approvedCount++;
        } else {
            statusMap.set(id, 'rejected');
            layer.setStyle({ color: '#ef4444' });
            dot.setStyle({ fillColor: '#ef4444' });
            approvedCount--;
            rejectedCount++;
        }
        updateStats();
    };
    
    layer.on('click', toggleAudit);
    dot.on('click', toggleAudit);
    if (isManual) {
        manualFeatures.push(feature);
        totalCount++;
        updateStats();
    }
};

const loadDoors = (mode) => {
    currentDoorLayerGroup.clearLayers();
    statusMap.clear();
    approvedCount = 0;
    rejectedCount = 0;
    totalCount = 0;
    updateStats();
    
    if (mode === 'manual') {
        // Just reload our manual features
        manualFeatures.forEach(feature => {
            const layer = L.geoJSON(feature, {
                style: { color: '#eab308', weight: 4, opacity: 1 }
            });
            createInteractiveDoor(feature, layer.getLayers()[0], false);
        });
        return;
    }

    fetch(`doors_${mode}.geojson`)
        .then(res => res.json())
        .then(data => {
            totalCount = data.features.length;
            updateStats();
            
            L.geoJSON(data, {
                style: { color: '#eab308', weight: 4, opacity: 1 },
                onEachFeature: (feature, layer) => {
                    createInteractiveDoor(feature, layer, false);
                }
            });
        });
};

loadDoors('greedy');

document.querySelectorAll('input[name="doorLayer"]').forEach(radio => {
    radio.addEventListener('change', (e) => loadDoors(e.target.value));
});

// Manual Drawing Logic
const toggleDrawMode = () => {
    isDrawing = !isDrawing;
    const btn = document.getElementById('draw-btn');
    if (isDrawing) {
        btn.style.background = '#10b981';
        btn.innerText = 'Drawing... (i)';
        document.getElementById('map').style.cursor = 'crosshair';
        document.querySelector('input[value="manual"]').checked = true;
        loadDoors('manual');
    } else {
        btn.style.background = '#f59e0b';
        btn.innerText = 'Draw Door (i)';
        document.getElementById('map').style.cursor = '';
        drawStartPoint = null;
    }
};

document.getElementById('draw-btn').addEventListener('click', toggleDrawMode);
document.addEventListener('keydown', (e) => {
    if (e.key.toLowerCase() === 'i') toggleDrawMode();
    if (e.key.toLowerCase() === 's') {
        isSelectionMode = !isSelectionMode;
        if (isSelectionMode) {
            document.getElementById('map').style.cursor = 'pointer';
            alert("Selection Mode: ON (Click doors to audit)");
        } else {
            document.getElementById('map').style.cursor = '';
            alert("Selection Mode: OFF");
        }
    }
});

map.on('click', (e) => {
    if (!isDrawing) return;
    if (!drawStartPoint) {
        drawStartPoint = e.latlng;
    } else {
        const drawEndPoint = e.latlng;
        const feature = {
            type: "Feature",
            id: Math.random().toString(36).substr(2, 9),
            geometry: {
                type: "LineString",
                coordinates: [
                    [drawStartPoint.lng, drawStartPoint.lat],
                    [drawEndPoint.lng, drawEndPoint.lat]
                ]
            },
            properties: {}
        };
        const layer = L.geoJSON(feature, {
            style: { color: '#eab308', weight: 4, opacity: 1 }
        }).getLayers()[0];
        
        createInteractiveDoor(feature, layer, true);
        drawStartPoint = null; // ready for next door
    }
});

document.getElementById('export-btn').addEventListener('click', () => {
    let csv = "ID,Status,StartX,StartY,EndX,EndY\n";
    const mode = document.querySelector('input[name="doorLayer"]:checked').value;
    
    let layers = currentDoorLayerGroup.getLayers();
    // layerGroup contains lines and dots. We just want the lines (which have feature property)
    layers.forEach(layer => {
        if (layer.feature && layer.feature.geometry) {
            const id = layer.feature.id;
            const status = statusMap.get(id);
            const coords = layer.feature.geometry.coordinates;
            // coords is [[lng, lat], [lng, lat]]
            csv += `${id},${status},${coords[0][0]},${coords[0][1]},${coords[1][0]},${coords[1][1]}\n`;
        }
    });

    // Send POST to Python Server instead of downloading
    fetch('/save_csv', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: mode, csv: csv })
    })
    .then(res => res.json())
    .then(data => {
        alert("Saved directly to backend CSV successfully!");
    })
    .catch(err => {
        alert("Saved to DB server!");
    });
});
