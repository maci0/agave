import { CloseIcon, DeleteIcon, NewIcon } from '../../ui/icons';
import { Button } from '../../ui/button';
import { cn } from '../../ui/cn';
import type { ReactNode } from 'react';
import type { ConvRecord } from '../types';

export type SidebarProps = {
  conversations: Array<ConvRecord> | null;
  loadError: string | null;
  /** A turn is in flight: the mutating actions wait for it. */
  streaming: boolean;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onNew: () => void;
  onRetryLoad: () => void;
  /** Hide the drawer-only close control. */
  onClose?: () => void;
};

/** The non-list states of the sidebar: loading, empty, and the retry after a
 *  failed fetch. */
const ListState = ({ children }: { children: ReactNode }) => (
  <div className="px-3 py-5 text-center font-mono text-xs text-faint">{children}</div>
);

const RetryState = ({ message, onRetry }: { message: string; onRetry: () => void }) => (
  <ListState>
    {message}
    <div className="mt-2.5">
      <Button type="button" variant="primaryOutline" size="sm" onClick={onRetry} aria-label="Retry loading conversations">
        Retry
      </Button>
    </div>
  </ListState>
);

/** One row: the title button and the delete control beside it. */
const ConversationRow = ({ conversation, streaming, onSelect, onDelete }: {
  conversation: ConvRecord;
  streaming: boolean;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}) => {
  const label = conversation.title ?? 'New chat';
  return (
    <div
      role="listitem"
      className={cn(
        'mb-0.5 flex items-center gap-1 rounded-md border border-transparent ps-3 pe-0.5 transition-colors hover:bg-muted',
        conversation.active === true && 'border-primary/25 bg-primary/10',
      )}
    >
      <button
        type="button"
        onClick={function () { onSelect(conversation.id); }}
        aria-label={label}
        aria-current={conversation.active === true ? 'true' : undefined}
        disabled={streaming}
        className="flex min-h-11 min-w-0 flex-1 items-center rounded-sm text-start"
      >
        <span
          className={cn('flex-1 truncate text-sm', conversation.active === true ? 'text-foreground' : 'text-muted-foreground')}
          title={conversation.title}
        >
          {label}
        </span>
      </button>
      <button
        type="button"
        onClick={function () { onDelete(conversation.id); }}
        disabled={streaming}
        aria-label={`Delete conversation: ${label}`}
        className="agave-reveal inline-flex size-11 shrink-0 items-center justify-center rounded-xs text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
      >
        <DeleteIcon className="size-4" aria-hidden="true" />
      </button>
    </div>
  );
};

const ConversationList = ({ conversations, loadError, streaming, onSelect, onDelete, onRetryLoad }: SidebarProps) => {
  if (loadError !== null) { return <RetryState message={loadError} onRetry={onRetryLoad} />; }
  if (conversations === null) { return <ListState>Loading conversations…</ListState>; }
  if (conversations.length === 0) {
    return (
      <ListState>
        No conversations yet.
        <br />
        Use <strong className="font-medium text-muted-foreground">New</strong> to start.
      </ListState>
    );
  }
  return (
    <div role="list" aria-label="Conversations">
      {conversations.map((conversation) =>
        (
          <ConversationRow
            key={conversation.id}
            conversation={conversation}
            streaming={streaming}
            onSelect={onSelect}
            onDelete={onDelete}
          />
        )
      )}
    </div>
  );
};

/** The conversation drawer body, shared by the desktop column and the mobile
 *  sheet so both surfaces stay identical.
 *
 *  The heading carries a fixed id so the mobile sheet can point
 *  `aria-labelledby` at it: inside the drawer this whole panel is the dialog's
 *  content, and the Radix dialog carried no title of its own, so a screen
 *  reader announced the sheet as an unnamed dialog. On the desktop column the
 *  dialog is not involved and `aside`'s `aria-label` names the landmark. */
export const Sidebar = (props: SidebarProps) => (
  <div className="flex h-full w-full flex-col bg-card">
    <div className="flex shrink-0 items-center justify-between gap-2 border-b border-divider px-3 py-3">
      <h2 id="sidebar-heading" className="font-mono text-xs font-medium text-faint">Chats</h2>
      <div className="flex shrink-0 items-center gap-1.5">
        <Button type="button" variant="primaryOutline" size="sm" onClick={props.onNew} disabled={props.streaming} aria-label="New conversation">
          <NewIcon className="size-4" aria-hidden="true" />
          New
        </Button>
        {props.onClose ? (
          <Button type="button" variant="plain" size="iconSm" onClick={props.onClose} aria-label="Close sidebar">
            <CloseIcon className="size-5" aria-hidden="true" />
          </Button>
        ) : null}
      </div>
    </div>
    <div className="agave-scroll flex-1 overflow-y-auto p-2">
      <ConversationList {...props} />
    </div>
  </div>
);