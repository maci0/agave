import { Image as ImageIcon, SlidersHorizontal, X } from 'lucide-react';
import { useCallback, useEffect, useRef, useState, type RefObject, type SyntheticEvent } from 'react';
import { Button } from '../../ui/button';
import { SettingsPanel } from './settings-panel';
import { cn } from '../../ui/cn';
import type { Sampling } from '../types';

const COMPOSER_MAX_HEIGHT_PX = 200;
const IMAGE_PREVIEW_MAX_PX = 80;

type ComposerProps = {
  sampling: Sampling;
  onSamplingChange: (next: Sampling) => void;
  onSubmit: (text: string, image: string | null) => void;
  streaming: boolean;
  onStop: () => void;
  vision: boolean;
  pendingImage: string | null;
  onImageFile: (file: File, label: string) => void;
  onRemoveImage: () => void;
  onClearSystem: () => void;
  /** Tokens per second, or null when idle. */
  tps: number | null;
  settingsOpen: boolean;
  onToggleSettings: () => void;
  /** Bumped by the app to pull focus back to the composer after a turn. */
  focusToken: number;
};

/** The attached image, with the control that removes it. */
const ImagePreview = ({ src, onRemove }: { src: string; onRemove: () => void }) => (
  <div className="mx-auto block w-full agave-measure p-2 max-drawer:px-0">
    <div className="relative inline-block max-w-full">
      <img
        className="block max-w-[200px] rounded-lg border border-border"
        style={{ maxHeight: `${IMAGE_PREVIEW_MAX_PX}px` }}
        src={src}
        alt="Attached image preview"
      />
      <Button
        type="button"
        size="iconSm"
        onClick={onRemove}
        aria-label="Remove image"
        title="Remove image"
        // The glyph uses --background on --destructive (5.98:1) and
        // On the lighter hover red (8.47:1). White measured only 2.99:1,
        // Below the 1.4.3 minimum for this label.
        className="absolute -end-2 -top-2 rounded-pill border-none bg-destructive text-background hover:bg-destructive-foreground"
      >
        <X className="size-4" aria-hidden="true" />
      </Button>
    </div>
  </div>
);

/** The message field: Enter to send, Shift+Enter for a new line, paste or drop
 *  an image, and a height that follows its content. */
type PromptFieldProps = ComposerProps & {
  text: string;
  onText: (next: string) => void;
  area: RefObject<HTMLTextAreaElement | null>;
};

const PromptField = (props: PromptFieldProps) => (
  <textarea
    ref={props.area}
    id="msg"
    name="message"
    rows={1}
    value={props.text}
    dir="auto"
    placeholder="Prompt"
    aria-label="Message input"
    aria-describedby="input-hint"
    enterKeyHint="send"
    autoComplete="off"
    autoFocus
    disabled={props.streaming}
    onChange={function (event) { props.onText(event.target.value); }}
    onKeyDown={function (event) {
      // Ignore Enter while an IME composition is active (CJK input).
      // There, Enter confirms the conversion; it must not send.
      if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
        event.preventDefault();
        event.currentTarget.form?.requestSubmit();
      }
    }}
    onPaste={function (event) {
      // Safari's DataTransferItemList is array-like without an iterator,
      // So it is copied to a real array before the scan.
      // oxlint-disable-next-line unicorn/no-useless-spread -- Safari has no iterator on this list
      for (const item of [...event.clipboardData.items]) {
        if (item.type.startsWith('image/')) {
          event.preventDefault();
          const pasted = item.getAsFile();
          if (pasted !== null) { props.onImageFile(pasted, 'Image pasted'); }
          return;
        }
      }
    }}
    // Min-width: 0 lets the field shrink below its intrinsic width.
    className="max-h-50 min-h-12 min-w-0 flex-1 resize-none rounded-lg border border-input bg-background px-4 py-3 leading-relaxed text-base text-foreground transition-[border-color,box-shadow] outline-none focus:border-primary focus:shadow-focus disabled:opacity-50 max-drawer:text-[16px]"
  />
);

/** The send key, which becomes the stop key while a turn is running. */
const SendControl = ({ streaming, canSend, onStop }: { streaming: boolean; canSend: boolean; onStop: () => void }) => (
  streaming ? (
    <Button type="button" variant="destructive" size="lg" onClick={onStop} title="Stop generation" aria-label="Stop generation">
      Stop
    </Button>
  ) : (
    <Button type="submit" variant="primaryOutline" size="lg" disabled={!canSend} title="Send message (Enter)" aria-label="Send message">
      Send
    </Button>
  )
);

/** The generation-speed readout, empty until a turn reports one. */
const SpeedReadout = ({ tps }: { tps: number | null }) => (
  tps === null ? null : (
    <span role="status" aria-label="Generation speed" className="flex items-center px-1 font-mono text-xs whitespace-nowrap text-primary">
      {`${tps.toFixed(1)} tok/s`}
    </span>
  )
);

/** The attach key, the settings key, and the send or stop control. */
const InputRow = (props: ComposerProps & {
  text: string;
  onText: (next: string) => void;
  canSend: boolean;
  area: RefObject<HTMLTextAreaElement | null>;
  fileInput: RefObject<HTMLInputElement | null>;
}) => (
  <div className="mx-auto flex w-full agave-measure items-end gap-2.5">
    <input
      ref={props.fileInput}
      type="file"
      accept="image/jpeg,image/png,image/gif,image/webp"
      className="hidden"
      aria-label="Attach image"
      onChange={function (event) {
        const file = event.target.files?.[0];
        // Clear the field so picking the same image again is still a change
        // Event: an unsupported or oversized file is often re-picked smaller.
        event.target.value = '';
        if (file !== undefined) { props.onImageFile(file, 'Image attached'); }
      }}
    />
    {props.vision ? (
      <Button
        type="button"
        size="icon"
        onClick={function () { props.fileInput.current?.click(); }}
        title="Attach image (or paste / drop)"
        aria-label="Attach image. You can also paste or drop an image."
        disabled={props.streaming}
      >
        <ImageIcon className="size-5" aria-hidden="true" />
      </Button>
    ) : null}
    <PromptField {...props} text={props.text} onText={props.onText} />
    <Button
      type="button"
      size="icon"
      active={props.settingsOpen}
      onClick={props.onToggleSettings}
      title="Sampling settings"
      aria-label={props.settingsOpen ? 'Close sampling settings' : 'Open sampling settings'}
      aria-expanded={props.settingsOpen}
      aria-controls="settings-panel"
    >
      <SlidersHorizontal className="size-5" aria-hidden="true" />
    </Button>
    <SendControl streaming={props.streaming} canSend={props.canSend} onStop={props.onStop} />
    <SpeedReadout tps={props.tps} />
  </div>
);

/** The form's own behavior: autosize, the drag-and-drop image path, and the
 *  focus pull the app requests after a turn. */
const useComposerForm = (props: ComposerProps, text: string, setText: (next: string) => void) => {
  const [dragOver, setDragOver] = useState(false);
  const area = useRef<HTMLTextAreaElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const focusedOnce = useRef(false);

  useEffect(function () {
    const element = area.current;
    if (!element) {return;}
    element.style.height = 'auto';
    element.style.height = `${Math.min(element.scrollHeight, COMPOSER_MAX_HEIGHT_PX)}px`;
  }, [text]);

  useEffect(function () {
    if (!focusedOnce.current) {
      focusedOnce.current = true;
      return;
    }
    area.current?.focus();
  }, [props.focusToken]);

  const onSubmit = useCallback(function (event: SyntheticEvent<HTMLFormElement>) {
    event.preventDefault();
    if (props.streaming) {return;}
    if (!text.trim() && props.pendingImage === null) {return;}
    props.onSubmit(text.trim(), props.pendingImage);
    setText('');
  }, [props, setText, text]);

  return { dragOver, setDragOver, area, fileInput, onSubmit };
};

export const Composer = (props: ComposerProps) => {
  const [text, setText] = useState('');
  const form = useComposerForm(props, text, setText);
  const canSend = Boolean(text.trim() || props.pendingImage) && !props.streaming;

  return (
    <form
      aria-label="Chat"
      onSubmit={form.onSubmit}
      onDragOver={function (event) {
        event.preventDefault();
        if (props.vision) { form.setDragOver(true); }
      }}
      onDragLeave={function (event) {
        // Ignore leave events that stay inside the form (child to child flicker).
        if (event.relatedTarget instanceof Node && event.currentTarget.contains(event.relatedTarget)) {return;}
        form.setDragOver(false);
      }}
      onDrop={function (event) {
        event.preventDefault();
        form.setDragOver(false);
        const [dropped] = event.dataTransfer.files;
        if (dropped.type.startsWith('image/')) { props.onImageFile(dropped, 'Image dropped'); }
      }}
      className={cn(
        'relative z-10 border-t border-border bg-card px-6 pt-4 pb-5 transition-colors max-drawer:p-4',
        form.dragOver && 'border-primary bg-primary/10',
      )}
    >
      {props.pendingImage !== null ? (
        <ImagePreview src={props.pendingImage} onRemove={props.onRemoveImage} />
      ) : null}

      {props.settingsOpen ? (
        <SettingsPanel sampling={props.sampling} onChange={props.onSamplingChange} onClearSystem={props.onClearSystem} />
      ) : null}

      <InputRow
        {...props}
        text={text}
        onText={setText}
        canSend={canSend}
        area={form.area}
        fileInput={form.fileInput}
      />
      <p id="input-hint" className="mx-auto mt-2 w-full agave-measure text-center font-mono text-2xs text-faint max-drawer:sr-only">
        Enter to send &middot; Shift+Enter for new line &middot; Escape to stop &middot; /help for commands
      </p>
    </form>
  );
};
