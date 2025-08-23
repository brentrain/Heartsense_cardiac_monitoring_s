
import React from 'react';
import { Patient, CardiacRhythm, PacerSettings } from '../types';
import CategorizedRhythmMenu from '@/components/CategorizedRhythmMenu';

interface RhythmControlPanelProps {
  selectedPatient: Patient | null;
  allRhythms: CardiacRhythm[];
  onSetRhythm: (rhythm: CardiacRhythm) => void;
  onUpdatePacerSettings: (settings: Partial<PacerSettings>) => void;
}

const RhythmControlPanel: React.FC<RhythmControlPanelProps> = ({
  selectedPatient,
  allRhythms,
  onSetRhythm,
  onUpdatePacerSettings,
}) => {
  if (!selectedPatient) {
    return (
      <div className="bg-slate-800 p-4 rounded-xl shadow-xl flex flex-col border border-slate-700/80">
        <h2 className="text-xl font-semibold text-sky-400 mb-3 border-b border-slate-700 pb-2">Rhythm & Pacing</h2>
        <p className="text-slate-400">Select a patient to control their cardiac rhythm and pacemaker settings.</p>
      </div>
    );
  }

  const currentPacerSettings = selectedPatient.pacerSettings || { mode: 'Off', rate: 70 };

  const handlePacerModeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    onUpdatePacerSettings({ mode: e.target.value as PacerSettings['mode'] });
  };

  const handlePacerRateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const rate = parseInt(e.target.value, 10);
    if (!isNaN(rate) && rate >= 30 && rate <= 250) {
      onUpdatePacerSettings({ rate });
    } else if (e.target.value === '') {
       onUpdatePacerSettings({ rate: 30 }); // Or some default minimum
    }
  };
  
  const handlePacerRateBlur = (e: React.ChangeEvent<HTMLInputElement>) => {
    let rate = parseInt(e.target.value, 10);
    if (isNaN(rate) || rate < 30) rate = 30;
    if (rate > 250) rate = 250;
    onUpdatePacerSettings({ rate });
  };


  return (
    <div className="bg-slate-800 p-4 rounded-xl shadow-xl flex flex-col border border-slate-700/80">
      <h2 className="text-xl font-semibold text-sky-400 mb-1 border-b border-slate-700 pb-2">
        Rhythm & Pacing
      </h2>
      <p className="text-sm text-slate-300 mb-3">
        Patient: <span className="font-semibold text-sky-400">{selectedPatient.name}</span>
      </p>
      
      <CategorizedRhythmMenu 
        allRhythms={allRhythms}
        activeRhythm={selectedPatient.activeRhythm}
        onSetRhythm={onSetRhythm}
      />

      {/* Pacemaker Settings */}
      <div className="border-t border-slate-700 pt-4 mt-4">
        <h3 className="text-md font-medium text-slate-200 mb-2">Pacemaker Settings</h3>
        <div className="space-y-3">
          <div>
            <label htmlFor="pacerMode" className="block text-sm font-medium text-slate-300">Mode</label>
            <select
              id="pacerMode"
              value={currentPacerSettings.mode}
              onChange={handlePacerModeChange}
              className="mt-1 block w-full bg-slate-700 border border-slate-600 rounded-md shadow-sm py-2 px-3 text-slate-100 focus:outline-none focus:ring-sky-500 focus:border-sky-500 sm:text-sm"
            >
              <option value="Off">Off</option>
              <option value="Demand">Demand (VVI)</option>
            </select>
          </div>
          {currentPacerSettings.mode !== 'Off' && (
            <div>
              <label htmlFor="pacerRate" className="block text-sm font-medium text-slate-300">Rate (bpm)</label>
              <input
                type="number"
                id="pacerRate"
                value={currentPacerSettings.rate.toString()}
                onChange={handlePacerRateChange}
                onBlur={handlePacerRateBlur}
                min="30"
                max="250"
                step="5"
                className="mt-1 block w-full bg-slate-700 border border-slate-600 rounded-md shadow-sm py-2 px-3 text-slate-100 focus:outline-none focus:ring-sky-500 focus:border-sky-500 sm:text-sm [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default RhythmControlPanel;