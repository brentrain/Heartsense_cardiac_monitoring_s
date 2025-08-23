/**
 * @file useCardiacDataSimulator.ts
 * @description
 * This custom React hook is the core of the HeartSenseAI simulation. It manages all patient data,
 * simulates real-time physiological data streams (ECG, vitals), handles alerts, and orchestrates
 * the HeartSenseAI predictive analysis feature.
 *
 * Key Responsibilities:
 * - State management for patients and their detailed simulation data.
 * - Persistence of patient lists and vital sign logs to localStorage.
 * - High-frequency simulation loop (ECG, waveforms) for visual fidelity.
 * - Low-frequency simulation loop (BP, Temp, alerts, vital logging) for periodic updates.
 * - Generation and management of clinical alerts based on configurable thresholds.
 * - Integration with a mock AI for the HeartSenseAI risk prediction feature.
 * - Provides an API for the rest of the application to interact with patient data (add, delete, update, etc.).
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { CardiacMetrics, ECGDataPoint, TrendDataPoint, Alert, AlertSeverity, Patient, PersonalizedAlertThresholds, CardiacRhythm, LoggedVitalSign, PacerSettings, PatientSpecificSimData, HeartSenseAIState, HeartSenseAIRiskLevel, HeartSenseAIOutput, AlarmFeedbackType, AlarmFeedbackEntry, PersistentPatientData, CardiacDataSimulator } from '../types';
import {
  SIMULATION_VITALS_INTERVAL_MS,
  SIMULATION_ECG_INTERVAL_MS,
  BP_UPDATE_INTERVAL_MS,
  TEMP_UPDATE_INTERVAL_MS,
  RR_UPDATE_INTERVAL_MS,
  AUTO_LOG_VITALS_INTERVAL_SECONDS,
  MAX_ECG_DATA_POINTS,
  MAX_HR_TREND_DATA_POINTS,
  ALERT_THRESHOLDS,
  generateUniqueId,
  getDefaultMetrics,
  getInitialEcgData,
  getInitialSpo2Data,
  getInitialRespData,
  RHYTHM_PARAMS,
  ECG_P_WAVE_PATTERN,
  ECG_PACER_SPIKE_PATTERN,
  ECG_PACED_QRS_PATTERN,
  ECG_SPO2_PULSE_PATTERN,
  ECG_SPO2_BASELINE_VALUE,
  SPO2_PULSE_POINTS_DURATION,
  RESP_AMPLITUDE,
  stretchPattern,
  INITIAL_HEARTSENSE_AI_STATE,
  HEARTSENSE_AI_ANALYSIS_INTERVAL_MS,
  HEARTSENSE_LOGGED_VITALS_COUNT,
  INITIAL_ALARM_FEEDBACK_LOG,
  INITIAL_PATIENTS
} from '../constants';

// --- HELPER FUNCTIONS ---

// Helper to generate random numbers within a specified range.
const generateRandomInRange = (min: number, max: number, decimalPlaces: number = 0): number => {
  const factor = Math.pow(10, decimalPlaces);
  return parseFloat((Math.random() * (max - min) + min).toFixed(decimalPlaces));
};


const createInitialPatientSimData = (patient: Patient): PatientSpecificSimData => {
    const rhythmParams = RHYTHM_PARAMS[patient.activeRhythm] || RHYTHM_PARAMS.NSR;
    const initialHR = rhythmParams.simulatedHR();
    return {
        metrics: getDefaultMetrics(initialHR),
        ecgData: getInitialEcgData(),
        spo2WaveData: getInitialSpo2Data(),
        respWaveData: getInitialRespData(),
        heartRateTrend: [],
        alerts: [],
        ecgPatternBuffer: [],
        ecgBufferIndex: 0,
        ecgCurrentTime: 0,
        wenckebachCycleBeat: 0,
        currentPRPoints: 0,
        mobitzBeatInCycle: 0,
        lastPWaveTime: 0,
        lastQRSWaveTime: 0,
        nextRRIntervalPoints: 0,
        pWaveCyclePosition: 0,
        acknowledgedAlertIds: new Set(),
        loggedVitals: [],
        timeSinceLastCardiacEventMs: 0,
        lastBeatTimestamp: 0,
        heartSenseAIState: { ...INITIAL_HEARTSENSE_AI_STATE },
        alarmFeedbackLog: [],
        currentTargetHeartRate: initialHR,
        lastBPUpdate: 0,
        lastTempUpdate: 0,
        lastRRUpdate: 0,
        lastSpo2UpdateAttempt: 0,
        nonNsrRhythmStartTime: patient.activeRhythm === 'NSR' ? null : Date.now(),
        spo2PulsePatternBuffer: [],
        spo2PulseBufferIndex: 0,
        spo2NextPulseDue: false,
        respirationCyclePosition: 0,
    };
};

/**
 * Simulates an AI analysis of patient data, returning a risk assessment.
 * This function replaces a real API call to provide functionality without needing an API key.
 * @param patient The patient being analyzed.
 * @param simData The patient's current simulation data.
 * @returns A promise that resolves to a `HeartSenseAIOutput` object.
 */
const mockRunHeartSenseAIAnalysis = async (patient: Patient, simData: PatientSpecificSimData): Promise<HeartSenseAIOutput> => {
    // Simulate network delay for realism
    await new Promise(resolve => setTimeout(resolve, generateRandomInRange(1500, 3000)));

    const { metrics, alerts } = simData;
    const rhythmParams = RHYTHM_PARAMS[patient.activeRhythm];

    let riskScore = generateRandomInRange(5, 20);
    let riskLevel = HeartSenseAIRiskLevel.STABLE;
    let predictionReasoning = "Patient vitals are stable and within normal parameters.";
    
    const unacknowledgedAlerts = alerts.filter(a => !simData.acknowledgedAlertIds.has(a.id));
    const hasCriticalAlerts = unacknowledgedAlerts.some(a => a.severity === AlertSeverity.CRITICAL);
    const hasWarningAlerts = unacknowledgedAlerts.some(a => a.severity === AlertSeverity.WARNING);

    // Rule-based logic to determine risk
    if (rhythmParams.isLethal) {
        riskScore = generateRandomInRange(90, 98);
        riskLevel = HeartSenseAIRiskLevel.CRITICAL;
        predictionReasoning = `Lethal rhythm (${rhythmParams.displayName}) detected, immediate intervention required.`;
    } else if (hasCriticalAlerts) {
        riskScore = generateRandomInRange(75, 90);
        riskLevel = HeartSenseAIRiskLevel.HIGH;
        const criticalAlert = unacknowledgedAlerts.find(a => a.severity === AlertSeverity.CRITICAL)!;
        predictionReasoning = `High risk due to critical alert for ${criticalAlert.metric}.`;
    } else if (patient.activeRhythm !== 'NSR' && !['SINUS_TACHYCARDIA', 'SINUS_BRADYCARDIA', 'FIRST_DEGREE_AV_BLOCK'].includes(patient.activeRhythm)) {
        riskScore = generateRandomInRange(50, 75);
        riskLevel = HeartSenseAIRiskLevel.MODERATE;
        predictionReasoning = `Moderate risk due to sustained abnormal rhythm (${rhythmParams.displayName}).`;
    } else if (hasWarningAlerts) {
        riskScore = generateRandomInRange(30, 50);
        riskLevel = HeartSenseAIRiskLevel.MODERATE;
        const warningAlert = unacknowledgedAlerts.find(a => a.severity === AlertSeverity.WARNING)!;
        predictionReasoning = `Moderate risk due to warning for ${warningAlert.metric}.`;
    } else if (patient.activeRhythm === 'SINUS_TACHYCARDIA' || metrics.heartRate > 100) {
        riskScore = generateRandomInRange(20, 40);
        riskLevel = HeartSenseAIRiskLevel.LOW;
        predictionReasoning = "Low risk, monitoring for sustained tachycardia is advised.";
    } else if (patient.activeRhythm === 'SINUS_BRADYCARDIA' || metrics.heartRate < 60) {
        riskScore = generateRandomInRange(20, 40);
        riskLevel = HeartSenseAIRiskLevel.LOW;
        predictionReasoning = "Low risk, monitoring for symptomatic bradycardia.";
    }

    if (simData.loggedVitals.length < HEARTSENSE_LOGGED_VITALS_COUNT / 2) {
        riskLevel = HeartSenseAIRiskLevel.INSUFFICIENT_DATA;
        riskScore = 0;
        predictionReasoning = "Insufficient historical data for a comprehensive analysis.";
    }
    
    return {
        riskScore,
        riskLevel,
        predictionReasoning,
        confidenceScore: generateRandomInRange(85, 98),
    };
};


export const useCardiacDataSimulator = (organizationName?: string | null): CardiacDataSimulator => {
    const [patients, setPatients] = useState<Patient[]>([]);
    const [selectedPatientId, setSelectedPatientId] = useState<string | null>(null);
    const [simulationDataMap, setSimulationDataMap] = useState(new Map<string, PatientSpecificSimData>());

    // --- INITIALIZATION ---
    useEffect(() => {
        // No API key initialization needed anymore.
    }, []);

    // --- DATA PERSISTENCE ---
    useEffect(() => {
        if (!organizationName) return;

        try {
            const storedData = localStorage.getItem(`cardiacMonitorData_${organizationName}`);
            if (storedData) {
                const data = JSON.parse(storedData);
                const loadedPatients = data.patients || INITIAL_PATIENTS;
                setPatients(loadedPatients);
                
                const newSimMap = new Map<string, PatientSpecificSimData>();
                for (const p of loadedPatients) {
                    const persistentData: PersistentPatientData = data.persistentPatientData?.[p.id] || { loggedVitals: [], alarmFeedbackLog: [] };
                    const simData = createInitialPatientSimData(p);
                    simData.loggedVitals = persistentData.loggedVitals || [];
                    simData.alarmFeedbackLog = persistentData.alarmFeedbackLog || [];
                     // Restore the persisted AI state to prevent re-analysis on every load
                    if (persistentData.heartSenseAIState) {
                        simData.heartSenseAIState = {
                            ...persistentData.heartSenseAIState,
                            isLoading: false, // Always reset loading state on app start
                            error: null,      // Always clear stale errors on app start
                        };
                    }
                    newSimMap.set(p.id, simData);
                }
                setSimulationDataMap(newSimMap);

                if (loadedPatients.length > 0) {
                     setSelectedPatientId(prevId => loadedPatients.some((p: Patient) => p.id === prevId) ? prevId : loadedPatients[0].id);
                } else {
                    setSelectedPatientId(null);
                }

            } else {
                setPatients(INITIAL_PATIENTS);
                const initialSimMap = new Map<string, PatientSpecificSimData>();
                INITIAL_PATIENTS.forEach(p => initialSimMap.set(p.id, createInitialPatientSimData(p)));
                setSimulationDataMap(initialSimMap);
                if (INITIAL_PATIENTS.length > 0) {
                    setSelectedPatientId(INITIAL_PATIENTS[0].id);
                }
            }
        } catch (e) {
            console.error("Failed to load data from localStorage", e);
            setPatients(INITIAL_PATIENTS); // Fallback
        }
    }, [organizationName]);

    useEffect(() => {
        if (!organizationName) return;
        try {
            const persistentPatientData: Record<string, PersistentPatientData> = {};
            simulationDataMap.forEach((simData, patientId) => {
                persistentPatientData[patientId] = {
                    loggedVitals: simData.loggedVitals,
                    alarmFeedbackLog: simData.alarmFeedbackLog,
                    heartSenseAIState: simData.heartSenseAIState, // Persist AI state
                };
            });
            const dataToStore = {
                patients,
                persistentPatientData,
            };
            localStorage.setItem(`cardiacMonitorData_${organizationName}`, JSON.stringify(dataToStore));
        } catch (e) {
            console.error("Failed to save data to localStorage", e);
        }
    }, [patients, simulationDataMap, organizationName]);


    const runHeartSenseAIAnalysis = useCallback(async (patient: Patient, simData: PatientSpecificSimData) => {
        if (simData.heartSenseAIState?.isLoading) return;

        setSimulationDataMap(prev => new Map(prev).set(patient.id, { ...simData, heartSenseAIState: { ...simData.heartSenseAIState!, isLoading: true } }));
        
        try {
            const aiOutput = await mockRunHeartSenseAIAnalysis(patient, simData);
            
            setSimulationDataMap(prev => {
                const currentSimData = prev.get(patient.id);
                if (!currentSimData) return prev;
                return new Map(prev).set(patient.id, {
                    ...currentSimData,
                    heartSenseAIState: {
                        ...aiOutput,
                        lastAnalyzedTimestamp: Date.now(),
                        isLoading: false,
                        error: null,
                    }
                });
            });
        } catch (error) {
            console.error("HeartSenseAI Mock Analysis Error:", error);
            setSimulationDataMap(prev => {
                const currentSimData = prev.get(patient.id);
                if (!currentSimData) return prev;
                const errorMessage = (error as Error).message || "Mock analysis failed.";

                return new Map(prev).set(patient.id, {
                    ...currentSimData,
                    heartSenseAIState: {
                        ...INITIAL_HEARTSENSE_AI_STATE,
                        riskLevel: HeartSenseAIRiskLevel.ERROR,
                        predictionReasoning: "An error occurred during analysis.",
                        lastAnalyzedTimestamp: Date.now(),
                        isLoading: false,
                        error: errorMessage,
                    }
                });
            });
        }
    }, []);


    // --- MAIN SIMULATION LOOPS ---
    useEffect(() => {
        // HIGH-FREQUENCY LOOP: ECG & WAVEFORMS
        const ecgIntervalId = setInterval(() => {
            const now = Date.now();
            setSimulationDataMap(prevMap => {
                const newMap = new Map(prevMap);
                patients.forEach(patient => {
                    let simData = newMap.get(patient.id);
                    if (!simData) return;

                    // Deep copy to prevent mutation issues, preserving the Set type
                    simData = { 
                        ...JSON.parse(JSON.stringify(simData)), 
                        acknowledgedAlertIds: new Set(simData.acknowledgedAlertIds) 
                    };

                    const rhythmParams = RHYTHM_PARAMS[patient.activeRhythm];
                    const pacerSettings = patient.pacerSettings || { mode: 'Off', rate: 70 };
                    const pacerOn = pacerSettings.mode !== 'Off';

                    let heartRate = simData.metrics.heartRate;
                    let nextBeatDue = false;
                    let isPacedBeat = false;
                    
                    const pointsPerSecond = 1000 / SIMULATION_ECG_INTERVAL_MS;
                    const rrIntervalPoints = heartRate > 0 ? Math.round((60 / heartRate) * pointsPerSecond) : Infinity;

                    if (simData.nextRRIntervalPoints <= 0) {
                        nextBeatDue = true;
                        if (pacerOn && heartRate < pacerSettings.rate) {
                            simData.currentTargetHeartRate = pacerSettings.rate;
                            isPacedBeat = true;
                        } else {
                            simData.currentTargetHeartRate = rhythmParams.simulatedHR();
                        }
                        const newRate = simData.currentTargetHeartRate;
                        simData.nextRRIntervalPoints = newRate > 0 ? (60 / newRate) * pointsPerSecond : Infinity;
                    }
                    simData.nextRRIntervalPoints--;


                    let currentPattern: number[] = [];
                    let patternIndex = simData.ecgBufferIndex;
                    
                    if (patternIndex > 0) {
                        currentPattern = simData.ecgPatternBuffer;
                    } else if (nextBeatDue) {
                        let basePattern = rhythmParams.basePattern;
                        // Handle Complex Rhythms
                        if(patient.activeRhythm === 'THIRD_DEGREE_AV_BLOCK'){
                            if(now - simData.lastPWaveTime > 60000 / rhythmParams.atrialRate!){
                                simData.ecgPatternBuffer = ECG_P_WAVE_PATTERN;
                                simData.lastPWaveTime = now;
                            } else {
                                basePattern = rhythmParams.basePattern;
                            }
                        }

                        // Pacing logic
                        if (isPacedBeat) {
                            currentPattern = [...ECG_PACER_SPIKE_PATTERN, ...ECG_PACED_QRS_PATTERN];
                        } else {
                            currentPattern = basePattern;
                        }

                        simData.ecgPatternBuffer = currentPattern;
                        simData.lastBeatTimestamp = now;
                        simData.spo2NextPulseDue = true; 
                    }


                    const newValue = currentPattern[patternIndex] || 0;
                    simData.ecgData = [...simData.ecgData.slice(1), { time: simData.ecgCurrentTime, value: newValue }];
                    simData.ecgCurrentTime++;

                    if (patternIndex >= currentPattern.length - 1) {
                        simData.ecgBufferIndex = 0;
                        simData.ecgPatternBuffer = [];
                    } else {
                        simData.ecgBufferIndex++;
                    }

                    // SpO2 Waveform
                    let spo2Value = ECG_SPO2_BASELINE_VALUE;
                    if(simData.spo2NextPulseDue && simData.spo2PulseBufferIndex === 0){
                        simData.spo2PulsePatternBuffer = ECG_SPO2_PULSE_PATTERN;
                    }
                    if(simData.spo2PulseBufferIndex < simData.spo2PulsePatternBuffer.length){
                        const pulseProgress = simData.spo2PulseBufferIndex / SPO2_PULSE_POINTS_DURATION;
                        const amplitude = (Math.sin(pulseProgress * Math.PI) * (simData.metrics.spo2 / 100)); // Modulate amplitude by spo2
                        spo2Value += simData.spo2PulsePatternBuffer[simData.spo2PulseBufferIndex] * 0.5 * Math.max(0.5, amplitude);
                        simData.spo2PulseBufferIndex++;
                    } else {
                        simData.spo2PulseBufferIndex = 0;
                        simData.spo2PulsePatternBuffer = [];
                        simData.spo2NextPulseDue = false;
                    }
                    simData.spo2WaveData = [...simData.spo2WaveData.slice(1), { time: simData.ecgCurrentTime, value: spo2Value }];
                    
                    // Respiration Waveform
                    const respRateHz = simData.metrics.respiratoryRate / 60;
                    const respCycleSeconds = 1 / respRateHz;
                    const respCyclePoints = respCycleSeconds * pointsPerSecond;
                    simData.respirationCyclePosition = (simData.respirationCyclePosition + 1) % respCyclePoints;
                    const respValue = Math.sin((simData.respirationCyclePosition / respCyclePoints) * 2 * Math.PI) * RESP_AMPLITUDE;
                    simData.respWaveData = [...simData.respWaveData.slice(1), { time: simData.ecgCurrentTime, value: respValue }];

                    newMap.set(patient.id, simData);
                });
                return newMap;
            });
        }, SIMULATION_ECG_INTERVAL_MS);

        // LOW-FREQUENCY LOOP: VITALS, ALERTS, LOGGING, AI
        const vitalsIntervalId = setInterval(() => {
            const now = Date.now();
            setSimulationDataMap(prevMap => {
                const newMap = new Map(prevMap);
                patients.forEach(patient => {
                    let simData = newMap.get(patient.id);
                    if (!simData) return;
                    
                    // Deep copy to avoid mutation issues, preserving the Set type
                    simData = { 
                        ...JSON.parse(JSON.stringify(simData)), 
                        acknowledgedAlertIds: new Set(simData.acknowledgedAlertIds) 
                    };
                    
                    const rhythmParams = RHYTHM_PARAMS[patient.activeRhythm];
                    
                    // Update HR based on target rate
                    const hrDiff = simData.currentTargetHeartRate - simData.metrics.heartRate;
                    simData.metrics.heartRate += Math.sign(hrDiff) * Math.min(Math.abs(hrDiff), 2);
                    
                    // Update other vitals periodically
                    if (now - (simData.lastBPUpdate || 0) > BP_UPDATE_INTERVAL_MS) {
                        simData.metrics.bloodPressure.systolic += generateRandomInRange(-3, 3);
                        simData.metrics.bloodPressure.diastolic += generateRandomInRange(-2, 2);
                        simData.lastBPUpdate = now;
                    }
                    if (now - (simData.lastTempUpdate || 0) > TEMP_UPDATE_INTERVAL_MS) {
                        simData.metrics.temperature += generateRandomInRange(-0.1, 0.1, 1);
                        simData.lastTempUpdate = now;
                    }
                    if (now - (simData.lastRRUpdate || 0) > RR_UPDATE_INTERVAL_MS) {
                        simData.metrics.respiratoryRate += generateRandomInRange(-1, 1);
                         simData.metrics.respiratoryRate = Math.max(8, Math.min(35, simData.metrics.respiratoryRate));
                        simData.lastRRUpdate = now;
                    }

                    // Lethal rhythm effects
                    if (rhythmParams.isLethal) {
                        simData.metrics.bloodPressure.systolic = 0;
                        simData.metrics.bloodPressure.diastolic = 0;
                        simData.metrics.spo2 = 0;
                        simData.metrics.respiratoryRate = 0;
                    }

                    // Check for new alerts
                    const checkAlert = (metric: string, value: number, thresholds: any, personalizedThresholds: any, severity: AlertSeverity, type: 'low' | 'high') => {
                        const gt = thresholds;
                        const pt = personalizedThresholds;
                        let threshold;
                        if (type === 'low') {
                            threshold = pt?.lowCritical ?? gt.LOW_CRITICAL;
                            if (severity === AlertSeverity.WARNING) threshold = pt?.lowWarning ?? gt.LOW_WARNING;
                        } else {
                            threshold = pt?.highCritical ?? gt.HIGH_CRITICAL;
                             if (severity === AlertSeverity.WARNING) threshold = pt?.highWarning ?? gt.HIGH_WARNING;
                        }
                        const isTriggered = type === 'low' ? value < threshold : value > threshold;
                        if (isTriggered) {
                            const message = `${metric} ${type === 'low' ? 'Low' : 'High'}: ${value}`;
                            if (!simData.alerts.some(a => a.message === message && a.severity === severity)) {
                                simData.alerts.push({ id: generateUniqueId(), message, severity, timestamp: new Date(), metric });
                            }
                        }
                    };
                    
                    checkAlert('Heart Rate', simData.metrics.heartRate, ALERT_THRESHOLDS.HEART_RATE, patient.personalizedThresholds?.heartRate, AlertSeverity.CRITICAL, 'low');
                    checkAlert('Heart Rate', simData.metrics.heartRate, ALERT_THRESHOLDS.HEART_RATE, patient.personalizedThresholds?.heartRate, AlertSeverity.WARNING, 'low');
                    checkAlert('Heart Rate', simData.metrics.heartRate, ALERT_THRESHOLDS.HEART_RATE, patient.personalizedThresholds?.heartRate, AlertSeverity.CRITICAL, 'high');
                    checkAlert('Heart Rate', simData.metrics.heartRate, ALERT_THRESHOLDS.HEART_RATE, patient.personalizedThresholds?.heartRate, AlertSeverity.WARNING, 'high');
                    checkAlert('SpO2', simData.metrics.spo2, ALERT_THRESHOLDS.SPO2, patient.personalizedThresholds?.spo2, AlertSeverity.CRITICAL, 'low');
                    checkAlert('SpO2', simData.metrics.spo2, ALERT_THRESHOLDS.SPO2, patient.personalizedThresholds?.spo2, AlertSeverity.WARNING, 'low');

                    // Log vitals
                    if (simData.loggedVitals.length === 0 || now - simData.loggedVitals[simData.loggedVitals.length-1].timestamp > AUTO_LOG_VITALS_INTERVAL_SECONDS * 1000) {
                        simData.loggedVitals.push({ timestamp: now, metrics: simData.metrics });
                        if(simData.loggedVitals.length > 200) simData.loggedVitals.shift(); // Keep log size reasonable
                    }

                    // Trigger AI Analysis
                    if (now - (simData.heartSenseAIState?.lastAnalyzedTimestamp || 0) > HEARTSENSE_AI_ANALYSIS_INTERVAL_MS) {
                        runHeartSenseAIAnalysis(patient, simData);
                    }
                    
                    newMap.set(patient.id, simData);
                });
                return newMap;
            });
        }, SIMULATION_VITALS_INTERVAL_MS);

        return () => {
            clearInterval(ecgIntervalId);
            clearInterval(vitalsIntervalId);
        };
    }, [patients, runHeartSenseAIAnalysis]);
    

    // --- PATIENT MANAGEMENT API ---
    const addPatient = useCallback((newPatientData: Omit<Patient, 'id' | 'activeRhythm' | 'pacerSettings' | 'personalizedThresholds' | 'noiseSignatures' | 'source'>, source = 'Manual Entry') => {
        const newPatient: Patient = {
            ...newPatientData,
            id: `PID${Date.now()}`,
            activeRhythm: 'NSR',
            pacerSettings: { mode: 'Off', rate: 70 },
            noiseSignatures: [],
            source,
        };
        const newSimData = createInitialPatientSimData(newPatient);
        
        setPatients(prev => [...prev, newPatient]);
        setSimulationDataMap(prev => new Map(prev).set(newPatient.id, newSimData));
        setSelectedPatientId(newPatient.id);
    }, []);

    const deletePatient = useCallback((patientId: string) => {
        // This function is designed to be robust by using functional state updates,
        // which prevents issues with stale state and ensures reliable deletion.

        // Functional update for the simulation data map
        setSimulationDataMap(prevMap => {
            const newMap = new Map(prevMap);
            // .delete returns true if an element was successfully removed
            if (!newMap.delete(patientId)) {
                console.warn(`Delete warning: Sim data for patient ID ${patientId} not found in map.`);
            }
            return newMap;
        });

        // Functional update for the patients list
        setPatients(prevPatients => {
            const patientIndex = prevPatients.findIndex(p => p.id === patientId);
            if (patientIndex === -1) {
                console.warn(`Delete failed: Patient with ID ${patientId} not found in list.`);
                return prevPatients; // Patient not found, do nothing to the list
            }

            const updatedPatients = prevPatients.filter(p => p.id !== patientId);

            // Functional update for the selected patient ID to ensure atomicity
            setSelectedPatientId(prevSelectedId => {
                // Only update selection if the deleted patient was the selected one
                if (prevSelectedId !== patientId) {
                    return prevSelectedId;
                }
                
                // If it was the selected patient, find the next one
                if (updatedPatients.length > 0) {
                    // Select the patient at the same index, or the new last patient
                    const newIndex = Math.min(patientIndex, updatedPatients.length - 1);
                    return updatedPatients[newIndex].id;
                }
                
                // No patients left
                return null;
            });

            return updatedPatients;
        });
    }, []); // No dependencies are needed as the function relies entirely on functional updates.


    const selectPatient = useCallback((patientId: string) => {
        setSelectedPatientId(patientId);
    }, []);

    const updatePatientDetails = useCallback((patientId: string, details: Partial<Patient>) => {
        setPatients(prev => prev.map(p => p.id === patientId ? { ...p, ...details } : p));
    }, []);

    const setPatientRhythm = useCallback((patientId: string, rhythm: CardiacRhythm) => {
        setPatients(prev => prev.map(p => {
            if (p.id === patientId) {
                return { ...p, activeRhythm: rhythm };
            }
            return p;
        }));
        setSimulationDataMap(prev => {
             const newMap = new Map(prev);
             const simData = newMap.get(patientId);
             if (simData) {
                 const newSimData = { ...simData, currentTargetHeartRate: RHYTHM_PARAMS[rhythm].simulatedHR() };
                 newMap.set(patientId, newSimData);
             }
             return newMap;
        });
    }, []);
    
    const acknowledgePatientAlerts = useCallback((patientId: string, alertTypeToAcknowledge?: AlertSeverity) => {
        setSimulationDataMap(prev => {
            const newMap = new Map(prev);
            const simData = newMap.get(patientId);
            if (simData) {
                const newAcknowledged = new Set(simData.acknowledgedAlertIds);
                simData.alerts.forEach(alert => {
                    if (!alertTypeToAcknowledge || alert.severity === alertTypeToAcknowledge) {
                        newAcknowledged.add(alert.id);
                    }
                });
                const newSimData = { ...simData, acknowledgedAlertIds: newAcknowledged };
                newMap.set(patientId, newSimData);
            }
            return newMap;
        });
    }, []);
    
    const updatePatientPacingSettings = useCallback((patientId: string, settings: Partial<PacerSettings>) => {
        setPatients(prev => prev.map(p => p.id === patientId ? { ...p, pacerSettings: { ...p.pacerSettings!, ...settings } } : p));
    }, []);

    const updatePatientNoiseSignatures = useCallback((patientId: string, signatures: string[]) => {
        setPatients(prev => prev.map(p => p.id === patientId ? { ...p, noiseSignatures: signatures } : p));
    }, []);

    const provideAlarmFeedback = useCallback((patientId: string, feedbackType: Omit<AlarmFeedbackType, 'not_reviewed'>) => {
         setSimulationDataMap(prev => {
            const newMap = new Map(prev);
            const simData = newMap.get(patientId);
            if (simData) {
                const newSimData = { 
                    ...JSON.parse(JSON.stringify(simData)), 
                    acknowledgedAlertIds: new Set(simData.acknowledgedAlertIds) 
                };
                
                const unacknowledgedWarningAlerts = newSimData.alerts
                    .filter((a: Alert) => a.severity === AlertSeverity.WARNING && !newSimData.acknowledgedAlertIds.has(a.id));

                unacknowledgedWarningAlerts.forEach((alert: Alert) => {
                    const feedbackEntry: AlarmFeedbackEntry = {
                        id: generateUniqueId(),
                        patientId,
                        alertSnapshot: {
                            message: alert.message,
                            severity: alert.severity,
                            metric: alert.metric,
                            originalTimestamp: new Date(alert.timestamp).getTime(),
                        },
                        feedback: feedbackType,
                        feedbackTimestamp: Date.now(),
                    };
                    newSimData.alarmFeedbackLog.push(feedbackEntry);
                    newSimData.acknowledgedAlertIds.add(alert.id);
                });

                newMap.set(patientId, newSimData);
            }
            return newMap;
        });
    }, []);


    return {
        patients,
        simulationDataMap,
        addPatient,
        deletePatient,
        selectPatient,
        selectedPatientId,
        updatePatientDetails,
        setPatientRhythm,
        acknowledgePatientAlerts,
        setPatients,
        updatePatientPacingSettings,
        updatePatientNoiseSignatures,
        provideAlarmFeedback,
    };
};