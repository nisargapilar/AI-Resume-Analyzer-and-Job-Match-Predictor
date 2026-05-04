import express from 'express';
import mongoose from 'mongoose';
import cors from 'cors';
import dotenv from 'dotenv';
import multer from 'multer';
import axios from 'axios';
import FormData from 'form-data';

dotenv.config();
const app = express();
const upload = multer({ storage: multer.memoryStorage() });

app.use(cors({ origin: 'http://localhost:5173' }));
app.use(express.json());

app.use((req, res, next) => {
  console.log(`${req.method} ${req.path}`);
  next();
});

mongoose.connect(process.env.MONGO_URI)
  .then(() => console.log('MongoDB connected'))
  .catch(err => console.error(err));

app.post('/api/analyze/text', async (req, res) => {
  try {
    const { resume_text, job_description } = req.body;
    const response = await axios.post(`${process.env.ML_SERVICE_URL}/analyze/text`, {
      resume_text,
      job_description
    });
    res.json(response.data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/analyze/pdf', upload.single('resume'), async (req, res) => {
  try {
    const form = new FormData();
    form.append('resume', req.file.buffer, {
      filename: req.file.originalname,
      contentType: 'application/pdf'
    });

    const jobDescription = req.body.job_description || req.query.job_description || '';

    const response = await axios.post(
      `${process.env.ML_SERVICE_URL}/analyze/pdf?job_description=${encodeURIComponent(jobDescription)}`,
      form,
      { headers: form.getHeaders() }
    );
    res.json(response.data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.listen(process.env.PORT, () =>
  console.log(`API running on port ${process.env.PORT}`)
);