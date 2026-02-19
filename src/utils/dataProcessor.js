import Papa from 'papaparse';
import * as ss from 'simple-statistics';

export const REQUIRED_COLUMNS = [
  'User_ID',
  'Product_ID',
  'Category',
  'Price (Rs.)',
  'Discount (%)',
  'Final_Price(Rs.)',
  'Payment_Method',
  'Purchase_Date'
];

export const parseCSV = (file) => {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      delimitersToGuess: [',', '\t', '|', ';'],
      transformHeader: (h) => h.trim(),
      complete: (results) => {
        const rawHeaders = (results.meta.fields || []).map(h => h.trim());

        // Find mapping: standard_name -> user_name
        const headerMap = {};
        REQUIRED_COLUMNS.forEach(req => {
          const found = rawHeaders.find(h => h.toLowerCase() === req.toLowerCase());
          if (found) headerMap[req] = found;
        });

        const missing = REQUIRED_COLUMNS.filter(col => !headerMap[col]);

        if (missing.length > 0) {
          reject(`Missing columns: ${missing.join(', ')}`);
          return;
        }

        const processedData = results.data.map(row => {
          // Robust number parsing (handle commas, scientific notation)
          const parseNum = (val) => {
            if (typeof val === 'number') return val;
            if (typeof val === 'string') {
              const clean = val.replace(/,/g, '');
              const parsed = parseFloat(clean);
              return isNaN(parsed) ? 0 : parsed;
            }
            return 0;
          };

          const date = new Date(row[headerMap['Purchase_Date']]);
          return {
            ...row,
            'User_ID': String(row[headerMap['User_ID']] || ''),
            'Product_ID': String(row[headerMap['Product_ID']] || ''),
            'Category': String(row[headerMap['Category']] || 'Other'),
            'Price (Rs.)': parseNum(row[headerMap['Price (Rs.)']]),
            'Discount (%)': parseNum(row[headerMap['Discount (%)']]),
            'Final_Price(Rs.)': parseNum(row[headerMap['Final_Price(Rs.)']]),
            'Payment_Method': String(row[headerMap['Payment_Method']] || 'Unknown'),
            'Purchase_Date': date,
            Month: date.toLocaleString('default', { month: 'short' }),
            Year: date.getFullYear(),
            MonthYear: `${date.getFullYear()}-${(date.getMonth() + 1).toString().padStart(2, '0')}`
          };
        }).filter(row => !isNaN(row.Purchase_Date.getTime()));

        if (processedData.length === 0) {
          reject("No valid records found. Check date format in 'Purchase_Date'.");
          return;
        }

        resolve(processedData);
      },
      error: (err) => reject(err.message)
    });
  });
};

// Smoothing Algorithms
export const smoothData = (data, type, windowSize = 3) => {
  const values = data.map(d => d.value);
  let smoothed = [];

  if (type === 'mean') {
    for (let i = 0; i < values.length; i++) {
      const start = Math.max(0, i - Math.floor(windowSize / 2));
      const end = Math.min(values.length, i + Math.floor(windowSize / 2) + 1);
      const subset = values.slice(start, end);
      smoothed.push(ss.mean(subset));
    }
  } else if (type === 'median') {
    for (let i = 0; i < values.length; i++) {
      const start = Math.max(0, i - Math.floor(windowSize / 2));
      const end = Math.min(values.length, i + Math.floor(windowSize / 2) + 1);
      const subset = values.slice(start, end);
      smoothed.push(ss.median(subset));
    }
  } else if (type === 'boundary') {
    // Simple boundary smoothing: snap to min/max of window
    for (let i = 0; i < values.length; i++) {
      const start = Math.max(0, i - Math.floor(windowSize / 2));
      const end = Math.min(values.length, i + Math.floor(windowSize / 2) + 1);
      const subset = values.slice(start, end);
      const min = Math.min(...subset);
      const max = Math.max(...subset);
      smoothed.push(Math.abs(values[i] - min) < Math.abs(values[i] - max) ? min : max);
    }
  } else {
    smoothed = values;
  }

  return data.map((d, i) => ({ ...d, smoothed: smoothed[i] }));
};

// Normalization
export const normalize = (values, type) => {
  if (type === 'min-max') {
    const min = Math.min(...values);
    const max = Math.max(...values);
    return values.map(v => (v - min) / (max - min));
  } else if (type === 'z-score') {
    const mean = ss.mean(values);
    const std = ss.standardDeviation(values);
    return values.map(v => (v - mean) / std);
  } else if (type === 'decimal-scaling') {
    const maxAbs = Math.max(...values.map(v => Math.abs(v)));
    const j = Math.ceil(Math.log10(maxAbs));
    return values.map(v => v / Math.pow(10, j));
  }
  return values;
};

// Binning
export const binFinalPrice = (price) => {
  if (price <= 1000) return 'Low';
  if (price <= 5000) return 'Medium';
  if (price <= 10000) return 'High';
  return 'Premium';
};
